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

from pathlib import Path
from contextlib import suppress, nullcontext
from copy import deepcopy
import csv
from typing import List, Dict, Any, Tuple, Optional

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

import torch_pruning as tp
import logging, sys


from train import get_args_parser, adjust_config, count_params, set_seed


logger = logging.getLogger("tp_exp")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


@torch.no_grad()
def measure_tp_latency(
    model: torch.nn.Module,
    loader,
    device: str = "cuda",
    multiple: int = 1,
    amp: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Return (throughput img/s, latency ms per image).
    - Moves model to `device` and sets eval().
    - Converts uint8 inputs to float.
    - Uses CUDA autocast(fp16) only on GPU when amp=True.
    """
    model = model.eval().to(device, non_blocking=True)

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    ac = torch.cuda.amp.autocast if (use_cuda and amp) else nullcontext

    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()

    num_images = 0
    for i in range(multiple):
        for bi, (images, *_) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                print('Reached max number of batches: ', max_batches)
                break

            images = images.to(device, non_blocking=True)

            # if not images.is_floating_point():
            #     images = images.float()

            with ac():
                _ = model(images)
            num_images += images.size(0)

            if bi % 100 == 0:
                print(f'{i} / {multiple}: {bi} / {len(loader)}')

    if use_cuda:
        torch.cuda.synchronize()
    elapsed = max(1e-8, time.time() - start)

    throughput = float(num_images / elapsed)             # img/s
    latency_ms = float((elapsed / num_images) * 1000.0)  # ms/img
    return throughput, latency_ms


@torch.no_grad()
def measure_perf(
    cost_metrics: Dict[str, float],
    model: torch.nn.Module,
    loader,
    tp_measure: str = "both",
    input_size: int = 224,
    device: str = "cuda",
    amp: bool = True,
    setting: str = "base",
) -> Dict[str, float]:
    """
    Populates:
      macs_{setting}, params_{setting},
      throughput_gpu_{setting}, latency_ms_gpu_{setting},
      throughput_cpu_{setting}, latency_ms_cpu_{setting}
    """

    # --- MACs / Params: use model's native device & dtype ---
    try:
        p0 = next(model.parameters())
        m_dev, m_dtype = p0.device, p0.dtype
    except StopIteration:
        # no params? fall back
        m_dev, m_dtype = torch.device("cpu"), torch.float32

    example_inputs = torch.randn(1, 3, input_size, input_size, device=m_dev, dtype=m_dtype)
    macs, params = tp.utils.count_ops_and_params(model.eval(), example_inputs)
    cost_metrics[f"macs_g_{setting}"] = float(macs) / 1e9
    cost_metrics[f"params_m_{setting}"] = float(params) / 1e6

    # --- GPU throughput/latency ---
    if tp_measure in ("gpu", "both") and torch.cuda.is_available():
        tp_gpu, lat_gpu = measure_tp_latency(
            model=model,
            loader=loader,
            device="cuda",
            multiple=3,
            amp=amp,
        )
        cost_metrics[f"throughput_gpu_{setting}"] = tp_gpu
        cost_metrics[f"latency_ms_gpu_{setting}"] = lat_gpu

    # --- CPU throughput/latency ---
    if tp_measure in ("cpu", "both"):
        m_cpu = deepcopy(model).to("cpu", non_blocking=True)
        # ensure fp32 on CPU (some pruned/ema models might carry half tensors)
        for p in m_cpu.parameters():
            if p.dtype != torch.float32:
                m_cpu = m_cpu.float()
                break

        m_dev, m_dtype = torch.device("cpu"), torch.float32
        example_inputs = torch.randn(1, 3, input_size, input_size, device=m_dev, dtype=m_dtype)

        tp_cpu, lat_cpu = measure_tp_latency(
            model=m_cpu,
            loader=loader,
            device="cpu",
            multiple=1,
            amp=False,  # AMP is CUDA-only
            max_batches=10,
        )
        cost_metrics[f"throughput_cpu_{setting}"] = tp_cpu
        cost_metrics[f"latency_ms_cpu_{setting}"] = lat_cpu

    return cost_metrics

def calc_importance(kind: str):
    if kind == 'l1':
        return tp.importance.GroupMagnitudeImportance(p=1)
    if kind == 'taylor':
        return tp.importance.GroupTaylorImportance()
    if kind == 'group_norm':
        return tp.importance.GroupNormImportance(p=2)
    return tp.importance.GroupMagnitudeImportance(p=2)  # l2


def populate_taylor_grads(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           dataset_is_multilabel: bool,
                           batches: int,
                           amp_autocast):
    """Accumulate grads for Taylor importance."""
    model.train()
    model.zero_grad(set_to_none=True)
    ce = torch.nn.BCEWithLogitsLoss() if dataset_is_multilabel else torch.nn.CrossEntropyLoss()
    it = iter(data_loader)
    for _ in range(max(1, batches)):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(data_loader); x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_autocast():
            logits = model(x)
            loss = ce(logits, y if dataset_is_multilabel else y.long())
        loss.backward()  # accumulate grads only; no optimizer step

def structured_prune(model: torch.nn.Module,
                      example_inputs: torch.Tensor,
                      ratio: float,
                      mode: str,
                      iter_steps: int,
                      ignored: List[torch.nn.Module],
                      round_to: int,
                      importance: str,
                      data_loader,
                      device,
                      dataset_is_multilabel: bool,
                      taylor_batches: int,
                      amp_autocast):
    logger.info(f'[Mode]: {mode} [Importance]: {importance}')
    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=calc_importance(importance),
        pruning_ratio=ratio,
        ignored_layers=ignored, round_to=round_to, global_pruning=True
    )
    if mode == 'oneshot':
        if importance == 'taylor':
            populate_taylor_grads(model, data_loader, device, dataset_is_multilabel, taylor_batches, amp_autocast)
        pruner.step()
    else:
        step_ratio = 1 - (1 - ratio) ** (1.0 / max(1, iter_steps))
        pruner = tp.pruner.BasePruner(
            model, example_inputs,
            importance=pruner.importance,
            pruning_ratio=step_ratio,
            ignored_layers=ignored, round_to=round_to, global_pruning=True
        )
        for _ in range(iter_steps):
            logger.info(f'[Iter Step]: {_}/{iter_steps}')
            if importance == 'taylor':
                populate_taylor_grads(model, data_loader, device, dataset_is_multilabel, taylor_batches, amp_autocast)
            pruner.step()
    model.zero_grad(set_to_none=True)



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
    pr = f'_{args.pruning_ratio}' if args.pruning_ratio else ''

    if setting:
        args.run_name = '{}_{}{}{}{}{}{}{}_{}_{}'.format(args.dataset_name, args.model, kr, head, clc, clr, fz, pr, args.setting, args.serial)
    else:
        args.run_name = '{}_{}{}{}{}{}{}{}_{}'.format(args.dataset_name, args.model, kr, head, clc, clr, fz, pr, args.serial)

    if args.output_dir and utils.is_main_process():
        args.output_dir = os.path.join(args.output_dir, args.run_name)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return 0


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
 
    # pruning
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--pruning_importance', type=str, default='l1')
    parser.add_argument('--pruning_mode', type=str, default='oneshot',
                        choices=['oneshot', 'iterative'])
    parser.add_argument('--pruning_steps', type=int, default=5)
    parser.add_argument('--pruning_taylor_batches', type=int, default=1)
    parser.add_argument('--tp-round-to', type=int, default=8,
                        help='round channels to multiples (kernel efficiency)')
    
    args = parser.parse_args()
    adjust_config(args)

    if args.plot_gradients:
        args.use_amp = False

    utils.init_distributed_mode(args)

    print('Main process and rank: ', utils.is_main_process(), utils.get_rank())
 
    args.total_batch_size = args.batch_size * args.grad_accum_steps * utils.get_world_size()

    set_run_name(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

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

            if args.random_resized_crop:
                t.append(transforms.RandomResizedCrop((input_size, input_size), interpolation=BICUBIC))
            else:
                # args.square_resize_random_crop
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


    if utils.is_main_process() and not args.eval and not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name

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

        if args.dataset_name.lower() != "imagenet":
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
        print(ret)
        
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
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=1000,
            global_pool='avg',
            args = args
        )
        
        if teacher_model is not None and args.dataset.lower() != "imagenet":
            teacher_model.reset_classifier(args.num_classes)

        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # Setup teacher model for DynamicViT distillation
    if "dyvit" in args.model:
        if args.no_dyvit_teacher:
            teacher_model = None
        else:          
            teacher_model = create_model(args.model+"_teacher",
                            pretrained = True,
                            num_classes=1000,
                            drop_rate=args.drop,
                            drop_path_rate=args.drop_path,
                            drop_block_rate=None,
                            img_size=args.input_size,
                            args = args
            )
            
            if args.dataset_name.lower() != "imagenet":
                assert(args.dyvit_teacher_weights != ""), "Empty DyViT Teacher Weight path"
                assert(os.path.isfile(args.dyvit_teacher_weights)), "Invalid DyViT Teacher Weight path: {}".format(args.dyvit_teacher_weights)

                teacher_model.reset_classifier(args.num_classes)
                
                checkpoint = torch.load(args.dyvit_teacher_weights, map_location='cpu')

                if checkpoint["ema_best"]:
                    teacher_model.load_state_dict(checkpoint['model_ema'])
                else:
                    teacher_model.load_state_dict(checkpoint['model'])

            teacher_model.to(device)
            teacher_model.eval()

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
        test_stats = evaluate(data_loader_val, model, device)
        if utils.is_main_process():
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

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


    ## pruning related
    # Baseline (post-pre) measurements
    # base_stats = evaluate(data_loader_val, model, device, amp_autocast)
    cost_metrics = {'acc_base': test_stats["acc1"]}

    logger.info(f'Measuring Performance')
    cost_metrics = measure_perf(cost_metrics, model, data_loader_train, 'both',
                             args.input_size, device, args.use_amp, 'base')

    # just use the same model
    model.eval()

    # Ignore final classifier
    ignored = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and getattr(m, 'out_features', None) == args.num_classes:
            ignored.append(m)

    # Prune
    logger.info('Starting Pruning')
    structured_prune(
        model,
        torch.randn(1, 3, args.input_size,args.input_size, device=device),
        args.pruning_ratio, args.pruning_mode, args.pruning_steps,
        ignored, args.tp_round_to, args.pruning_importance,
        data_loader_train, device, False, args.pruning_taylor_batches, amp_autocast
    )

    # After-prune MACs/params
    cost_metrics = measure_perf(cost_metrics, model, data_loader_train, 'both',
                             args.input_size, device, args.use_amp, 'pruned')

    logger.info('Pruning flow finished.')


    # regular training
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

    total_time = time.time() - start_time
    time_total = round(total_time / 60, 2)  # mins
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    max_memory = round(max_memory, 4)

    no_params = count_params(model)
    no_params = round(no_params / (1e6), 4)  # millions of parameters

    no_params_trainable = count_params(model, trainable=True)
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions of parameters

    profiler = 'thop' if 'ats' in args.model else 'torchprofile'
    flops = utils.count_flops(model, args.input_size, device, profiler=profiler)
    flops = round(flops / 1e9, 4)
    print('FLOPs: ', flops)

    test_acc = test_stats['acc1']
    cost_metrics['acc_pruned'] = test_stats['acc1']
    cost_metrics = {k: round(v, 4) for k, v in cost_metrics.items()}

    if utils.is_main_process() and args.plot_gradients:
        gradients = pd.DataFrame.from_dict(gradient_list)
        fp = os.path.join(args.output_dir, 'gradients.csv')
        gradients.to_csv(fp, header=True, index=False)            

    if utils.is_main_process():
        print('Training time {}'.format(total_time_str))

        print('Pruning metrics: ', cost_metrics)

        if not args.debugging:
            wandb.log(cost_metrics, step=total_step)

            wandb.run.summary
            wandb.run.summary['test_acc'] = test_acc
            wandb.run.summary['best_acc'] = max_accuracy
            wandb.run.summary['time_total'] = time_total
            wandb.run.summary['flops'] = flops
            wandb.run.summary['max_memory'] = max_memory
            wandb.run.summary['no_params'] = no_params
            wandb.run.summary['no_params_trainable'] = no_params_trainable
            wandb.finish()

    return 0


if __name__ == '__main__':
    main()
    # parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()
    # adjust_config(args)
    # main(args)