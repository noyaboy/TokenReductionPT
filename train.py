# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import pandas as pd
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import yaml
from copy import deepcopy

from pathlib import Path
from contextlib import suppress

import wandb
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, AsymmetricLossMultiLabel
from scheduler_factory import create_scheduler
from optim import create_optimizer
from timm.utils import get_state_dict, ModelEmaV2
from mp_scaler import NativeScalerGradAcum
from torch.utils.data import default_collate

from datasets import build_dataset, MEANS, STDS
from engine import train_one_epoch, evaluate_multiclass, evaluate_multilabel
from losses import DistillationLoss, DynamicViTDistillationLoss, GroupLassoLoss
from samplers import RASampler

import utils
import models


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', '--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=45, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--scale-lr', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=200, help='eval every x epochs')
    parser.add_argument('--log_freq', type=int, default=100)

    # Model parameters
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model', '--model_name', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', '--image_size', default=224, type=int, help='images input size')
    parser.add_argument('--resize-size', type=int, default=None,
                        help='square resize size before resizing to input-size')
    parser.add_argument('--test-input-size', default=None, type=int, help='images input size')
    parser.add_argument('--test-resize-size', type=int, default=None,
                        help='square resize size before resizing to input-size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--try_fused_attn', action='store_false',
                        help='by def use fused kernel attention')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--transform_timm', action='store_true',
                        help='use timm transform function')
    parser.add_argument('--pad_random_crop', action='store_true',
                        help='pad to keep aspect ratio then crop')
    parser.add_argument('--short_side_resize_random_crop', action='store_true',
                        help='resize first so short side is resize_size then random crop a square')
    parser.add_argument('--random_resized_crop', action='store_true')
    parser.add_argument('--horizontal_flip', action='store_false')
    parser.add_argument('--rand_aug', action='store_true', help='use RandAugment')
    parser.add_argument('--trivial_aug', action='store_true', help='use trivialaugmentwide')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.3 in TR, 0.4 in CNX)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=False)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--re_prob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--re_mult', type=int, default=1, metavar='PCT',
                        help='Random erase applications (def: 1)')
    parser.add_argument('--re_size_min', default=0.02, type=float, help='min erasing area')
    parser.add_argument('--re_size_max', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--try_mixup_prefetch', action='store_true')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', '--ckpt_path', default='', help='finetune from checkpoint')
    parser.add_argument('--ft_resize_pos_embed', action='store_true',
                        help='interpolate pos embedding for vits with new image size')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    # parser.add_argument('--dataset_root_path', default='../../data/cub/CUB_200_2011', type=str,
    #                    help='dataset path')
    # parser.add_argument('--dataset_name', default='imagenet', choices=['imagenet', 'nabirds', "coco", "nuswide"],
    #                    type=str, help='Image Net dataset path')
    # parser.add_argument('--inat-category', default='name',
    #                    choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
    #                    type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='results_train',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', '--cpu_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.add_argument('--dataset_in_memory', action='store_true',
                        help='load whole dataset into RAM to reduce reads from storage')
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--transform_gpu', action='store_true')
    parser.add_argument('--dataset_image_folder', action='store_true', help='use tv imagefolder')

    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')

    parser.add_argument("--wandb_project", default="TokenReductionPT", type=str)
    parser.add_argument("--wandb_group", default="nycu_pcs", type=str)

    parser.add_argument('--backbone_lr_scale', default=1.0, type=float, help="")
    parser.add_argument('--backbone_freeze_steps', default=0, type=int, help="")
    parser.add_argument('--constant_cls', action='store_true', help="")
    parser.add_argument('--constant_pos', action='store_true', help="")


    parser.add_argument('--use_amp', '--fp16', action='store_false', help="")
    parser.add_argument('--sched_in_steps', action='store_true', help="")
    parser.add_argument('--grad_accum_steps', default = 1, type=int, help="")
    parser.add_argument('--lr_batch_normalizer', default = 1024, type=float, help="")
    
    parser.add_argument('--save_more_than_best', action='store_true', help="")

    temp_args, _ = parser.parse_known_args()
    
    parser.add_argument('--reduction_loc', type=int, nargs='+', default=[])
    parser.add_argument('--keep_rate', type=float, nargs='+', default=[])

    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')

    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='images',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='images',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='images',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')

    parser.add_argument('--train_trainval', action='store_true',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')
    parser.add_argument("--cfg", type=str, default='configs/soylocal_ft_weakaugs.yaml',
                        help="If using it overwrites args and reads yaml file in given path")

    parser.add_argument('--serial', type=int, default=420)
    parser.add_argument('--debugging', action='store_true')
    parser.add_argument('--plot_gradients', action='store_true')
    parser.add_argument('--save_images_wandb', action='store_true')

    # parameter-efficient evaluation
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--unfreeze_cls', action='store_true')
    parser.add_argument('--unfreeze_positional_embedding', action='store_true')
    parser.add_argument('--unfreeze_encoder_first_block', action='store_true')

    # modifications for fgir: data-augmentation, ifa head, reweight intermediate cls tokens
    parser.add_argument('--dropped_token_fusion', action='store_true')
    parser.add_argument('--fusion_crit_norm', action='store_true')
    parser.add_argument('--compression_factor', type=int, default=[64])
    parser.add_argument('--select_crit', type=str, nargs='+', default=[])
    parser.add_argument('--mcrit_choices', type=str, nargs='+', default=['maws', 'dmaws', 'attn', 'gls'])

    parser.add_argument('--ifa_head', action='store_true')
    parser.add_argument('--ifa_dws_conv_groups', type=int, default=2)
    parser.add_argument('--spatio_layer_agg', type=str, default=None)

    parser.add_argument('--clc', action='store_true', help='cross layer aggregation')
    parser.add_argument('--clc_include_gap', action='store_false')
    parser.add_argument('--clc_pool_cls', action='store_true')
    parser.add_argument('--clc_pool_clr', action='store_true')
    parser.add_argument('--clc_recover_at_last', action='store_false')
    parser.add_argument('--num_clr', type=int, default=0)

    parser.add_argument('--lasso_loss_weight', type=float, default=0.0000,
                        help='Toe2Head Lasso Regularization, def: 0.0001 from VQT')
    parser.add_argument('--lasso_inner_dim', type=int, default=0)
    parser.add_argument('--num_query_tokens', type=int, default=5,
                        help='Visual Query Tuning')

    # tr cnns (asea)
    parser.add_argument('--axis_start', type=str, default='-1',
                        help='start with last, -1, axis corresp to row red')
    parser.add_argument('--axis_squeeze', type=str, default='mean')
    parser.add_argument('--squeeze_proj', action='store_false')
    parser.add_argument('--axis_conv', type=int, default=3)
    parser.add_argument('--axis_norm', action='store_false')
    parser.add_argument('--red_dw_channels', type=int, default=1)
    parser.add_argument('--excitation_proj', action='store_false')
    parser.add_argument('--proj_v', action='store_false')

    parser.add_argument('--asea_pos', type=str, default='post_block',
                        choices=['parallel', 'parallelplus', 'pre_block',
                                 'in_block', 'post_block', 'post_blockplus'])
    parser.add_argument('--dropped_axis_fusion', action='store_false')

    parser.add_argument('--keep_rate_var', type=float, nargs='+', default=[])
    parser.add_argument('--red_train_only', action='store_true')
    parser.add_argument('--red_random_perturb', type=float, default=0.0)
    parser.add_argument('--asea_drop_path', type=float, default=0.0)
    parser.add_argument('--asea_ls_init_values', type=float, default=1e-5,
                        help='for layerscale in asea')
    parser.add_argument('--asea_disable_drop_path', action='store_true',
                        help='if true then disables drop path when asea is active')
 
    if "dyvit" in temp_args.model.lower():
        parser.add_argument('--token_distill_weight', default=0.5, type=float)
        parser.add_argument('--cls_distill_weight', default=0.5, type=float)
        parser.add_argument('--ratio_weight', default=2.0, type=float)
        parser.add_argument('--cls_weight', default=1.0, type=float)   
        parser.add_argument('--mse_token', action='store_true') 
        parser.add_argument('--dyvit_distill', action='store_true') 
        parser.add_argument('--no_dyvit_teacher', action='store_false') 
        parser.add_argument('--dyvit_teacher_weights', default="", type=str)
        parser.set_defaults(dyvit_distill=False)
        parser.set_defaults(mse_token=True)

    if "dpcknn" in temp_args.model.lower():
        parser.add_argument('--k_neighbors', default=5, type=int)
        
    if "heuristic" in temp_args.model.lower():
        parser.add_argument('--heuristic_pattern', type=str, default="l1", choices={"l1", "l2", "linf"})                        
        parser.add_argument('--min_radius', type=float, default=1.0) 
        parser.add_argument('--not_contiguous', action='store_true') 
    
    if "sinkhorn" in temp_args.model.lower():
        parser.add_argument('--sinkhorn_eps', type=float, default=1.0)   
        
    if "kmedoids" in temp_args.model.lower() or "sinkhorn" in temp_args.model.lower():
        parser.add_argument('--cluster_iters', type=int, default=3)   
        
    if "kmedoids" in temp_args.model.lower() or "dpcknn" in temp_args.model.lower():
        parser.add_argument('--equal_weight', action='store_true') 

    return parser


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            fp = cfg.get("defaults").get(d)
            cf = os.path.join(os.path.dirname(config_file), fp)
            with open(cf) as f:
                val = yaml.safe_load(f)
                print(val)
                cfg.update(val)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def adjust_config(args):
    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)


def count_params(model, trainable=False):
    if trainable:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    return sum([p.numel() for p in model.parameters()])


def cost_metrics(model_without_ddp, model, device, input_size, model_name):
    profiler = 'thop' if 'ats' in model_name else 'torchprofile'
    flops = utils.count_flops(model_without_ddp, input_size, device, profiler=profiler)
    flops = round(flops / 1e9, 4)
    print('FLOPs: ', flops)

    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    max_memory = round(max_memory, 4)

    no_params = count_params(model)
    no_params = round(no_params / (1e6), 4)  # millions of parameters

    no_params_trainable = count_params(model, trainable=True)
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions of parameters
    return flops, max_memory, no_params, no_params_trainable


def set_seed(seed):
    # fix the seed for reproducibility
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    return 0


def set_run_name(args, setting=False):
    args.keep_rate_single = args.keep_rate[0] if args.keep_rate else None
    kr = f'_{args.keep_rate_single}' if args.keep_rate_single else ''
    if args.ifa_head and args.ifa_dws_conv_groups:
        head = '_cla'
    elif args.ifa_head:
        head = '_ifa'
    else:
        head = ''
    clc = '_clc' if args.clc else ''
    clr = f'_{args.num_clr}' if args.num_clr else ''
    fz = '_fz' if args.freeze_backbone else ''

    if setting:
        args.run_name = '{}_{}{}{}{}{}{}_{}_{}'.format(args.dataset_name, args.model, kr, head, clc, clr, fz, args.setting, args.serial)
    else:
        args.run_name = '{}_{}{}{}{}{}{}_{}'.format(args.dataset_name, args.model, kr, head, clc, clr, fz, args.serial)

    if args.output_dir and utils.is_main_process():
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return 0


def build_model(args):
    print(f"Creating model: {args.model}")
    if 'vit' in args.model or 'deit' in args.model:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            pretrained_cfg=None,
            pretrained_cfg_overlay=None,
            num_classes=1000,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size,
            args = args
        )

        if args.dataset_name.lower() not in ['imagenet', 'imagenet1k'] or args.ifa_head:
            model.reset_classifier(args.num_classes)
        if args.num_clr:
            model.add_clr(args.num_clr)
    else:
        try:
            model = create_model(
                args.model,
                pretrained=args.pretrained,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=args.num_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                args=args,
            )
        except:
            model = create_model(
                args.model,
                pretrained=args.pretrained,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=args.num_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
            )

    print(model)
    return model


def build_teacher_model(args, device):
    args = deepcopy(args)
    args.keep_rate = [1.0]

    teacher_model = None

    if args.distillation_type != 'none':
        print(f"Creating teacher model: {args.teacher_model}")

        if 'vit' in args.teacher_model or 'deit' in args.teacher_model:
            teacher_model = create_model(
                args.teacher_model,
                pretrained=args.pretrained,
                pretrained_cfg=None,
                pretrained_cfg_overlay=None,
                num_classes=1000,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=args.input_size,
                args = args
            )

            if args.dataset_name.lower() not in ['imagenet', 'imagenet1k'] or args.ifa_head:
                teacher_model.reset_classifier(args.num_classes)
            if args.num_clr:
                teacher_model.add_clr(args.num_clr)
        else:
            try:
                teacher_model = create_model(
                    args.teacher_model,
                    pretrained=args.pretrained,
                    pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    num_classes=args.num_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    args=args,
                )
            except:
                teacher_model = create_model(
                    args.teacher_model,
                    pretrained=args.pretrained,
                    pretrained_cfg=None,
                    pretrained_cfg_overlay=None,
                    num_classes=args.num_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
            )

        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        elif args.teacher_path:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        else:
            checkpoint = None

        if checkpoint:
            ret = teacher_model.load_state_dict(checkpoint['model'], strict=False)
            print('Finetuned teacher checkpoint: ', ret)

        teacher_model.to(device)
        teacher_model.eval()

    return teacher_model


def main(args):
    if args.plot_gradients:
        args.use_amp = False

    utils.init_distributed_mode(args)

    print('Main process and rank: ', utils.is_main_process(), utils.get_rank())
 
    args.total_batch_size = args.batch_size * args.grad_accum_steps * utils.get_world_size()

    set_run_name(args)

    device = torch.device(args.device)

    set_seed(args.seed)

    cudnn.benchmark = True

    # adjust input sizes
    args.resize_size = args.resize_size if args.resize_size else int(args.input_size / 0.875)
    args.test_input_size = args.test_input_size if args.test_input_size else args.input_size
    args.test_resize_size = args.test_resize_size if args.test_resize_size else int(args.test_input_size / 0.875)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if utils.is_main_process():
        setattr(args, 'num_images_train', len(dataset_train))
        setattr(args, 'num_images_val', len(dataset_val))
        setattr(args, 'num_classes', args.num_classes)
        print(args.dataset_name, args.num_classes, len(dataset_train), len(dataset_val))

    if args.distributed and not args.dataset_in_memory:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    args.mixup_prefetch = False

    if mixup_active:
        if args.try_mixup_prefetch:
            try:
                from torchvision.transforms import v2
                args.mixup_prefetch = True
            except:
                args.mixup_prefetch = False

        if args.mixup_prefetch:
            mixup = v2.MixUp(alpha=args.mixup, num_classes=args.num_classes)
            cutmix = v2.CutMix(alpha=args.cutmix, num_classes=args.num_classes)

            mixup_or_cutmix = v2.RandomApply([v2.RandomChoice(
                [mixup, cutmix], 
                p=[args.mixup_switch_prob, 1-args.mixup_switch_prob]
            )], p=args.mixup_prob)

            def collate_fn(batch):
                return mixup_or_cutmix(*default_collate(batch))

        else:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)


    transform_fn = None
    if args.transform_gpu:
        try:
            from torchvision.transforms import v2 as transforms
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC

            t = []

            input_size = args.input_size

            if args.short_side_resize_random_crop:
                # already added as part of the cpu transform
                pass
            elif args.random_resized_crop:
                t.append(transforms.RandomResizedCrop((input_size, input_size), interpolation=BICUBIC))
            else:
                t.append(transforms.RandomCrop((input_size, input_size)))

            if args.horizontal_flip:
                t.append(transforms.RandomHorizontalFlip())

            if args.rand_aug:
                t.append(transforms.RandAugment())
            if args.trivial_aug:
                t.append(transforms.TrivialAugmentWide())

            mean = MEANS['imagenet']
            std = STDS['imagenet']
            if args.custom_mean_std:
                mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
                std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

            t.append(transforms.ToDtype(torch.float32, scale=True))
            t.append(transforms.Normalize(mean=mean, std=std, inplace=True))

            if args.re_prob > 0:
                t.append(
                    transforms.RandomErasing(
                        p=args.re_prob, scale=(args.re_size_min, args.re_size_max),
                        ratio=(args.re_r1, 3.3), inplace=True
                    )
                )
            transform_fn = transforms.Compose(t)
            print('Using GPU augmentation: ', transform_fn)
        except:
            raise NotImplementedError("--transform_gpu requires torchvision v2 transforms")


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        collate_fn=collate_fn if args.mixup_prefetch else default_collate,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    if utils.is_main_process() and not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name

    model = build_model(args)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if args.ft_resize_pos_embed:
            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

        ret = model.load_state_dict(checkpoint_model, strict=False)
        print('Finetuned checkpoint: ', ret)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            pass
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            pass

    if args.freeze_backbone:
        keywords = ['head', 'gate', 'clr']

        if args.unfreeze_cls:
            keywords.append('class_token')
            keywords.append('cls_token')
        if args.unfreeze_positional_embedding:
            keywords.append('embeddings.patch_embeddings')
            keywords.append('pos_embed')
        if args.unfreeze_encoder_first_block:
            keywords.append('blocks.0')

        unfrozen = []

        for name, param in model.named_parameters():
            if any(kw in name for kw in keywords):
                param.requires_grad = True
                unfrozen.append(name)
            else:
                param.requires_grad = False

        print(unfrozen)

        print('Total parameters (M): ', count_params(model) / (1e6))
        print('Trainable parameters (M): ', count_params(model, trainable=True) / (1e6))
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if utils.is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

    # Scale LR and create Optimizer
    args.num_steps_epoch = len(data_loader_train) * utils.get_world_size() // args.grad_accum_steps
    
    if args.scale_lr:
        linear_scaled_lr = args.lr * args.total_batch_size / args.lr_batch_normalizer
        args.input_lr = args.lr
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)

    # Setup 16-bit training if chosen
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.use_amp:
        loss_scaler = NativeScalerGradAcum()
        amp_autocast = torch.cuda.amp.autocast

    # Create LR Scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Create Loss function
    if args.dataset_name.lower() != "coco" and args.dataset_name.lower() != "nuswide":
        criterion = LabelSmoothingCrossEntropy()
        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        print("Using ASL Loss")
        criterion = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False)
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Setup teacher_model for Deit Distillation
    teacher_model = build_teacher_model(args, device)

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    if args.lasso_loss_weight:
        criterion = GroupLassoLoss(criterion, args.lasso_loss_weight, args.lasso_inner_dim)
    elif "dyvit" in args.model:
        criterion = DynamicViTDistillationLoss(
            criterion, teacher_model, args.ratio_weight, args.cls_distill_weight, args.token_distill_weight, args.cls_weight, args.mse_token)
    else:
        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
        )

    print(args)

    if args.dataset_name.lower() == "coco" or args.dataset_name.lower() == "nuswide":
        evaluate = evaluate_multilabel
    else:
        evaluate = evaluate_multiclass

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)

    if args.eval:
        start_time = time.time()
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        if utils.is_main_process():
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

            max_accuracy = test_stats["acc1"]
            log_stats = {**{f'val_{k}': v for k, v in test_stats.items()},
                            "max_accuracy": max_accuracy}

            total_time = time.time() - start_time
            time_mins = round(total_time / 60, 2)  # mins

            flops, max_memory, no_params, no_params_trainable = cost_metrics(
                model_without_ddp, model, device, args.input_size, args.model)

            if not args.debugging:
                wandb.log(log_stats, step=0)
                wandb.run.summary['test_acc'] = max_accuracy
                wandb.run.summary['best_acc'] = max_accuracy
                wandb.run.summary['time_total'] = time_mins
                wandb.run.summary['flops'] = flops
                wandb.run.summary['max_memory'] = max_memory
                wandb.run.summary['no_params'] = no_params
                wandb.run.summary['no_params_trainable'] = no_params_trainable
                wandb.finish()

        return 0

    test_stats = evaluate(data_loader_val, model, device, amp_autocast)
    
    if utils.is_main_process():
        max_accuracy = test_stats["acc1"]
        log_stats = {**{f'val_{k}': v for k, v in test_stats.items()},
                        "max_accuracy": max_accuracy}
        
        if model_ema is not None and not args.model_ema_force_cpu:
            log_stats = {**log_stats,
                         **{f'ema_val_{k}': v for k, v in test_stats.items()}}
        status_print = f"Epoch: 0\tDataset: {len(dataset_val)}\tAcc@1: {test_stats['acc1']:.1f}%"
        print(status_print)       

        if not args.debugging:
            wandb.log(log_stats, step=0)      

    total_epochs = args.epochs+args.cooldown_epochs

    if utils.is_main_process():
        print(f"Start training for {total_epochs} epochs")
        if not args.debugging:
            wandb.watch(model)

    start_time = time.time()
    max_accuracy = 0.0

    gradient_list = []

    if total_epochs == 0:
        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': 0,
                    'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                    'args': args,
                    'ema_best': False,
                }, checkpoint_path)
    else:                
        for epoch in range(args.start_epoch, total_epochs):

            for param_group in optimizer.param_groups:
                if epoch == 0:
                    for p in param_group["params"]:
                        p.grad =  None
                if epoch == param_group['fix_step']:
                    for p in param_group["params"]:
                        p.grad = torch.zeros_like(p)
        
            if hasattr(data_loader_train.sampler, 'set_epoch'):
                data_loader_train.sampler.set_epoch(epoch)

            train_stats, total_step = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, total_epochs, loss_scaler, lr_scheduler,
                args.clip_grad, model_ema, mixup_fn,
                transform_fn, amp_autocast,
                set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
                grad_accum_steps = args.grad_accum_steps,
                num_steps_epoch = len(data_loader_train),
                print_freq = args.log_freq,
                multi_label = args.dataset_name.lower() == "coco" or args.dataset_name.lower() == "nuswide",
                plot_gradients=args.plot_gradients, gradient_list=gradient_list,
                save_images_wandb=args.save_images_wandb, output_dir=args.output_dir,
                debugging=args.debugging
            )

            lr_scheduler.step(epoch+1)
            if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)
                
            if (epoch % args.eval_freq == 0 or epoch == total_epochs - 1) and epoch != 0:
                test_stats = evaluate(data_loader_val, model, device, amp_autocast)
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                test_stats1 = {f'test_{k}': v for k, v in test_stats.items()}
            else:
                test_stats1 = {}
            
            max_accuracy_flag = False

            if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                    }, checkpoint_path)

            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                max_accuracy_flag = True
                print(f'Max accuracy: {max_accuracy:.2f}%')
                if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                    checkpoint_paths = [output_dir / 'best_standard_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                            'args': args,
                        }, checkpoint_path)


            ema_test_states = None
            if model_ema is not None and not args.model_ema_force_cpu and (epoch % args.eval_freq == 0 or epoch == total_epochs - 1) and epoch != 0:
                ema_test_states = evaluate(data_loader_val, model_ema.module, device, amp_autocast)
            
                if max_accuracy < ema_test_states["acc1"]:
                    max_accuracy = ema_test_states["acc1"]
                    max_accuracy_flag = True
                    if args.output_dir and utils.is_main_process() and args.save_more_than_best:
                        checkpoint_paths = [output_dir / 'best_ema_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'model_ema': get_state_dict(model_ema),
                                'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                                'args': args,
                            }, checkpoint_path)

            if max_accuracy_flag:
                ema_best = False
                if model_ema is not None and ema_test_states is not None:
                    ema_best = ema_test_states["acc1"] > test_stats["acc1"]

                if args.output_dir and utils.is_main_process():
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                            'args': args,
                            'ema_best': ema_best,
                        }, checkpoint_path)

            if (epoch == total_epochs - 1) and args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'last_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
                        'args': args,
                        'ema_best': ema_best,
                    }, checkpoint_path)

            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in test_stats1.items()},
                            "max_accuracy": max_accuracy}
                status_print = f"Epoch: {epoch}\tDataset: {len(dataset_val)}\tAcc@1: {test_stats['acc1']:.1f}%"

                if ema_test_states is not None:
                    status_print += f"\tEMA-Acc@1: {ema_test_states['acc1']:.1f}%"
                    log_stats = {**log_stats,
                                **{f'ema_val_{k}': v for k, v in ema_test_states.items()}}
                                
                status_print += f"\tMax Acc@1: {max_accuracy:.2f}%"
                # print(status_print)       

                if not args.debugging:
                    wandb.log(log_stats, step=total_step)
                else:
                    break
      
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    if utils.is_main_process():
        total_time = time.time() - start_time
        time_mins = round(total_time / 60, 2)  # mins
        print('Training time {}'.format(time_mins))

        flops, max_memory, no_params, no_params_trainable = cost_metrics(
            model_without_ddp, model, device, args.input_size, args.model)

        test_acc = test_stats['acc1']

        if args.plot_gradients:
            gradients = pd.DataFrame.from_dict(gradient_list)
            fp = os.path.join(args.output_dir, 'gradients.csv')
            gradients.to_csv(fp, header=True, index=False)

        if not args.debugging:
            wandb.run.summary['test_acc'] = test_acc
            wandb.run.summary['best_acc'] = max_accuracy
            wandb.run.summary['time_total'] = time_mins
            wandb.run.summary['flops'] = flops
            wandb.run.summary['max_memory'] = max_memory
            wandb.run.summary['no_params'] = no_params
            wandb.run.summary['no_params_trainable'] = no_params_trainable
            wandb.finish()

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    adjust_config(args)
    main(args)
