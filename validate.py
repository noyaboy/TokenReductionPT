import argparse
import os
import numpy as np
import time
import torch
import yaml
import cv2

from json import JSONEncoder
from contextlib import suppress
from collections import OrderedDict
import utils

from timm.models import create_model
from timm.utils import accuracy, AverageMeter

from datasets import build_dataset

import models

import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data', metavar="DIR", type=str, help='dataset path')
parser.add_argument('--dataset', '-d', metavar='NAME', default='aircraft', choices=['imagenet', 'nabirds', "coco", "nuswide"], type=str, help='Dataset to evaluate on')
parser.add_argument('--split', metavar='NAME', default='validation', help='dataset split (default: validation)')

# 新增的資料集相關參數
parser.add_argument('--dataset_name', default='aircraft', type=str, help='dataset name')
parser.add_argument('--dataset_root_path', type=str, default='../../data/aircraft/fgvc-aircraft-2013b/data',
                    help='the root directory for where the data/feature/label files are')
parser.add_argument('--folder_test', type=str, default='images',
                    help='the directory where images are stored, ex: dataset_root_path/test/')
parser.add_argument('--df_test', type=str, default='test.csv',
                    help='the df csv with img_dirs, targets, root/test.csv')
parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                    help='the df csv with classnames and class ids, root/classid_classname.csv')
parser.add_argument('--train_trainval', action='store_true',
                    help='when true uses trainval for train and evaluates on test \
                    otherwise use train for train and evaluates on val')
parser.add_argument('--transform_gpu', action='store_true')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
parser.add_argument('--resize-size', type=int, default=None,
                    help='square resize size before resizing to input-size')

# add
parser.add_argument('--test-input-size', default=224, type=int, help='images input size')
parser.add_argument('--test-resize-size', type=int, default=224, help='square resize size before resizing to input-size')

parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use_amp', action='store_true', help="")
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

# 新增的模型相關參數
parser.add_argument('--reduction_loc', type=int, nargs='+', default=[]) 
parser.add_argument('--keep_rate', type=float, nargs='+', default=[])
parser.add_argument('--ifa_head', action='store_true')
parser.add_argument('--ifa_dws_conv_groups', type=int, default=2)
parser.add_argument('--clc', action='store_true', help='cross layer aggregation')
parser.add_argument('--clc_include_gap', action='store_false')
parser.add_argument('--clc_pool_cls', action='store_true')
parser.add_argument('--clc_recover_at_last', action='store_false')
parser.add_argument('--num_clr', type=int, default=0)
parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')
parser.add_argument('--cfg', type=str, default=None, help='YAML config file path')

parser.add_argument('--viz_mode', action='store_true', help="")
parser.add_argument('--viz_output_name', default='viz_results.json', help='visualisation output filename')
parser.add_argument('--viz_image_dir', default='viz_images', help='directory to save visualization images')
parser.add_argument('--viz_save_format', default='png', choices=['png', 'jpg', 'pdf'], help='format to save visualization images')
parser.add_argument('--viz_max_images', default=10, type=int, help='maximum number of images to visualize')
parser.add_argument('--viz_features', action='store_true', help='visualize feature maps')
parser.add_argument('--viz_attn', action='store_true', help='visualize attention maps')
parser.add_argument('--viz_tokens', action='store_true', help='visualize token positions')
parser.add_argument('--viz_overlay', action='store_true', help='visualize overlays on original images')
parser.add_argument('--viz_patch_size', type=int, default=16, help='patch size for token visualization')
parser.add_argument('--viz_cmap', type=str, default='hot', help='colormap for attention visualization')
parser.add_argument('--viz_highlight_dropped', action='store_true', help='highlight dropped tokens instead of kept tokens')

parser.add_argument('--model', default='', type=str, help='Model name (if not using checkpoint)')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')

parser.add_argument('--finetune', '--ckpt_path', default='', help='finetune from checkpoint')
parser.add_argument('--transform_timm', action='store_true', help='use timm transform function')

parser.add_argument('--dataset_in_memory', action='store_true',
                    help='load whole dataset into RAM to reduce reads from storage')
parser.add_argument('--prefetch_factor', type=int, default=4)
parser.add_argument('--dataset_tv', action='store_true', help='use tv imagefolder')

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

# 新增: 创建可视化目录
def create_viz_dir(args):
    if args.viz_mode:
        viz_dir = os.path.join(args.output_dir, args.viz_image_dir)
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        return viz_dir
    return None

# 新增: 可视化特征图函数
def visualize_feature_map(feature_map, save_path, cmap='viridis', title=None):
    """
    将特征图可视化为图像
    
    Args:
        feature_map (np.ndarray): 特征图数据，形状为 [H, W]
        save_path (str): 保存路径
        cmap (str): 颜色映射
        title (str): 图像标题
    """
    plt.figure(figsize=(8, 8))
    
    # 归一化特征图以便更好地可视化
    if np.max(feature_map) != np.min(feature_map):
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    
    plt.imshow(feature_map, cmap=cmap)
    plt.colorbar()
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 3. 修改可视化注意力图函数，添加更多控制参数
def visualize_attention_map(attn_map, save_path, title=None, cmap='hot'):
    """
    将注意力图可视化为热力图，改进版
    
    Args:
        attn_map (np.ndarray): 注意力图数据，形状为 [H, W]
        save_path (str): 保存路径
        title (str): 图像标题
        cmap (str): 颜色映射，默认为'hot'
    """
    plt.figure(figsize=(8, 8))
    
    # 检查attn_map是否包含有效数据
    if np.all(np.isnan(attn_map)) or np.all(attn_map == 0):
        print(f"Warning: Attention map contains all NaN or zeros.")
        plt.text(0.5, 0.5, "Invalid attention map", 
                 ha='center', va='center', fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # 归一化注意力图
    if np.max(attn_map) != np.min(attn_map):
        attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
    
    # 使用预定义的颜色映射或自定义颜色映射
    if cmap == 'attention_cmap':
        colors = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
        cm = LinearSegmentedColormap.from_list('attention_cmap', colors, N=100)
    else:
        cm = plt.get_cmap(cmap)
    
    plt.imshow(attn_map, cmap=cm)
    plt.colorbar()
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 4. 添加一个可视化注意力图叠加在原始图像上的函数
def visualize_attention_on_image(img, attn_map, save_path, title=None, alpha=0.6):
    """
    将注意力图叠加在原始图像上进行可视化
    
    Args:
        img (np.ndarray): 原始图像，形状为 [H, W, C]
        attn_map (np.ndarray): 注意力图数据，形状为 [H, W]
        save_path (str): 保存路径
        title (str): 图像标题
        alpha (float): 注意力图的透明度，范围[0,1]
    """
    plt.figure(figsize=(8, 8))
    
    # 检查attn_map是否包含有效数据
    if np.all(np.isnan(attn_map)) or np.all(attn_map == 0):
        print(f"Warning: Attention map contains all NaN or zeros.")
        plt.imshow(img)
        if title:
            plt.title(title + " (Invalid attention map)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # 确保注意力图的尺寸与图像匹配
    if attn_map.shape != img.shape[:2]:
        h, w = img.shape[:2]
        attn_map = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 归一化注意力图
    if np.max(attn_map) != np.min(attn_map):
        attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
    
    # 首先显示原始图像
    plt.imshow(img)
    
    # 使用'hot'颜色映射并设置透明度
    plt.imshow(attn_map, cmap='hot', alpha=alpha)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 1. 修复visualize_token_positions函数，更好地处理不同模型的patch大小
def visualize_token_positions(img_size, token_positions, save_path, title=None, model_type=None, highlight_kept=False):
    """
    可视化token的位置，改进版
    
    Args:
        img_size (tuple): 原始图像尺寸 (H, W)
        token_positions (np.ndarray): token位置索引
        save_path (str): 保存路径
        title (str): 图像标题
        model_type (str, optional): 模型类型，用于决定patch大小
        highlight_kept (bool): 如果为True，高亮保留的token；如果为False，高亮丢弃的token
    """
    h, w = img_size
    
    # 创建一个空白图像
    token_map = np.zeros((h, w), dtype=np.float32)
    
    # 确保token_positions是numpy数组
    if not isinstance(token_positions, np.ndarray):
        token_positions = np.array(token_positions)
    
    # 处理无效索引
    if len(token_positions) == 0 or (token_positions < 0).all():
        print(f"Warning: No valid token indices for visualization. Original indices: {token_positions}")
        plt.figure(figsize=(8, 8))
        plt.imshow(token_map, cmap='binary')
        plt.title(f"{title} (No valid tokens)" if title else "No valid tokens")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # 过滤无效的索引值
    valid_indices = token_positions[token_positions >= 0]
    
    if len(valid_indices) == 0:
        print(f"Warning: No valid token indices for visualization. Original indices: {token_positions}")
        plt.figure(figsize=(8, 8))
        plt.imshow(token_map, cmap='binary')
        plt.title(f"{title} (No valid tokens)" if title else "No valid tokens")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # 根据模型类型确定patch大小
    patch_size = 16  # 默认使用16x16的patch
    if model_type and 'vit_tiny' in model_type:
        patch_size = 16
    elif model_type and 'vit_small' in model_type:
        patch_size = 16
    elif model_type and 'vit_base' in model_type:
        patch_size = 16
    elif model_type and 'vit_large' in model_type:
        patch_size = 16
    
    # 如果索引值超出预期范围，尝试调整patch大小
    max_token_idx = np.max(valid_indices)
    expected_tokens = (h * w) // (patch_size * patch_size)
    
    if max_token_idx >= expected_tokens:
        # 如果索引超出范围，尝试较小的patch大小
        potential_patch_sizes = [16, 14, 8, 4]
        for ps in potential_patch_sizes:
            expected_tokens = (h * w) // (ps * ps)
            if max_token_idx < expected_tokens:
                patch_size = ps
                break
    
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w
    
    print(f"Visualization info: Image size={h}x{w}, patch size={patch_size}, "
          f"grid={num_patches_h}x{num_patches_w}, valid tokens={len(valid_indices)}, "
          f"max token idx={np.max(valid_indices)}, total patches={total_patches}")
    
    # 创建所有token的位置索引
    all_token_indices = np.arange(total_patches)
    
    # 根据highlight_kept参数决定要高亮显示哪些token
    if highlight_kept:
        # 高亮显示保留的token（原来的逻辑）
        highlight_indices = valid_indices
    else:
        # 高亮显示丢弃的token（新的逻辑）
        # 找出所有未在valid_indices中的索引
        highlight_indices = np.array([idx for idx in all_token_indices if idx not in valid_indices])
    
    # 将高亮的token标记为1
    for pos in highlight_indices:
        if pos < total_patches:  # 确保索引在有效范围内
            row = (pos // num_patches_w) * patch_size
            col = (pos % num_patches_w) * patch_size
            if row < h and col < w:  # 确保在图像范围内
                token_map[row:min(row+patch_size, h), col:min(col+patch_size, w)] = 1
    
    plt.figure(figsize=(8, 8))
    
    # 使用适当的颜色映射
    cmap = 'binary'
    if not highlight_kept:
        # 可以为丢弃的token使用不同的颜色
        cmap = 'Reds'
    
    plt.imshow(token_map, cmap=cmap, alpha=0.7)
    
    if title:
        # 根据高亮模式调整标题
        if not highlight_kept:
            if 'Token Position' in title:
                title = title.replace('Token Position', 'Dropped Token Position')
        plt.title(title)
    
    # 添加网格线以更清晰地显示patches
    ax = plt.gca()
    # 主要网格线（每个patch的边界）
    for i in range(0, h+1, patch_size):
        ax.axhline(i-0.5, color='gray', linestyle='-', alpha=0.3)
    for j in range(0, w+1, patch_size):
        ax.axvline(j-0.5, color='gray', linestyle='-', alpha=0.3)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Add visualization function that overlays the original image
def visualize_token_positions_on_image(img, token_positions, save_path, title=None, model_type=None, highlight_kept=False):
    """
    Visualize the positions of tokens on the original image

    Args:
    img (np.ndarray): original image, shape [H, W, C]
    token_positions (np.ndarray): token position index
    save_path (str): save path
    title (str): image title
    model_type (str, optional): model type, used to determine patch size
    highlight_kept (bool): If True, highlight the retained tokens; if False, highlight the discarded tokens
    """
    h, w = img.shape[0], img.shape[1]
    
    # Make sure token_positions is a numpy array
    if not isinstance(token_positions, np.ndarray):
        token_positions = np.array(token_positions)
    
    # Handling invalid indexes
    if len(token_positions) == 0 or (token_positions < 0).all():
        print(f"Warning: No valid token indices for visualization. Original indices: {token_positions}")
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"{title} (No valid tokens)" if title else "No valid tokens")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # Filter invalid index values
    valid_indices = token_positions[token_positions >= 0]
    
    if len(valid_indices) == 0:
        print(f"Warning: No valid token indices for visualization. Original indices: {token_positions}")
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"{title} (No valid tokens)" if title else "No valid tokens")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return
    
    # 根据模型类型确定patch大小
    patch_size = 16  # 默认使用16x16的patch
    if model_type and 'vit_tiny' in model_type:
        patch_size = 16
    elif model_type and 'vit_small' in model_type:
        patch_size = 16
    elif model_type and 'vit_base' in model_type:
        patch_size = 16
    elif model_type and 'vit_large' in model_type:
        patch_size = 16
    
    # 如果索引值超出预期范围，尝试调整patch大小
    max_token_idx = np.max(valid_indices)
    expected_tokens = (h * w) // (patch_size * patch_size)
    
    if max_token_idx >= expected_tokens:
        # 如果索引超出范围，尝试较小的patch大小
        potential_patch_sizes = [16, 14, 8, 4]
        for ps in potential_patch_sizes:
            expected_tokens = (h * w) // (ps * ps)
            if max_token_idx < expected_tokens:
                patch_size = ps
                break
    
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    total_patches = num_patches_h * num_patches_w
    
    print(f"Visualization info (overlay): Image size={h}x{w}, patch size={patch_size}, "
          f"grid={num_patches_h}x{num_patches_w}, valid tokens={len(valid_indices)}, "
          f"total patches={total_patches}")
    
    # 创建所有token的位置索引
    all_token_indices = np.arange(total_patches)
    
    # 根据highlight_kept参数决定要高亮显示哪些token
    if highlight_kept:
        # 高亮显示保留的token（原来的逻辑）
        highlight_indices = valid_indices
    else:
        # 高亮显示丢弃的token（新的逻辑）
        # 找出所有未在valid_indices中的索引
        highlight_indices = np.array([idx for idx in all_token_indices if idx not in valid_indices])
    
    # 创建一个遮罩层
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 将高亮的token标记为1
    for pos in highlight_indices:
        if pos < total_patches:  # 确保索引在有效范围内
            row = (pos // num_patches_w) * patch_size
            col = (pos % num_patches_w) * patch_size
            if row < h and col < w:  # 确保在图像范围内
                mask[row:min(row+patch_size, h), col:min(col+patch_size, w)] = 1
    
    # 创建一个彩色的遮罩层
    colored_mask = np.zeros((h, w, 4), dtype=np.float32)
    # 使用红色高亮丢弃的token，蓝色高亮保留的token
    if highlight_kept:
        colored_mask[..., 2] = 1.0  # 蓝色（保留的token）
    else:
        colored_mask[..., 0] = 1.0  # 红色（丢弃的token）
    colored_mask[..., 3] = mask * 0.4  # 40%的透明度
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.imshow(colored_mask)
    
    if title:
        # 根据高亮模式调整标题
        if not highlight_kept:
            if 'Token' in title and 'Overlay' in title:
                title = title.replace('Token Overlay', 'Dropped Token Overlay')
        plt.title(title)
    
    # 添加网格线以更清晰地显示patches
    ax = plt.gca()
    # 主要网格线（每个patch的边界）
    for i in range(0, h+1, patch_size):
        ax.axhline(i-0.5, color='white', linestyle='-', alpha=0.3)
    for j in range(0, w+1, patch_size):
        ax.axvline(j-0.5, color='white', linestyle='-', alpha=0.3)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 新增: 重构2D特征图
def reconstruct_feature_map(tokens, token_indices, h, w):
    """
    根据token索引重构2D特征图
    
    Args:
        tokens (np.ndarray): token特征，形状为 [num_tokens]
        token_indices (np.ndarray): token位置索引
        h, w (int): 输出特征图尺寸
    
    Returns:
        np.ndarray: 重构的2D特征图，形状为 [h, w]
    """
    feature_map = np.zeros((h, w), dtype=np.float32)
    
    patch_size = int(np.sqrt(h * w / len(token_indices)))
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size
    
    for idx, pos in enumerate(token_indices):
        if pos < num_patches_h * num_patches_w:  # 忽略CLS token等特殊token
            row = (pos // num_patches_w) * patch_size
            col = (pos % num_patches_w) * patch_size
            feature_map[row:row+patch_size, col:col+patch_size] = tokens[idx]
    
    return feature_map

def validate(args, _logger):
    amp_autocast = suppress  # do nothing
    if args.use_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')

    # 修改這個檢查，允許直接使用模型名稱而不是檢查點
    use_checkpoint = False
    use_direct_model = False
    
    if args.checkpoint and args.checkpoint != "":
        use_checkpoint = True
        assert os.path.isdir(args.checkpoint), "Checkpoint path is not dir, not usable: {}".format(args.checkpoint)
        assert os.path.isfile(os.path.join(args.checkpoint, "best_checkpoint.pth")), "Checkpoint path does not have a 'best_checkpoint.pth' file"
    elif args.model and args.model != "":
        use_direct_model = True
        _logger.info(f"Using direct model: {args.model} with pretrained={args.pretrained}")
    else:
        raise ValueError("Either --checkpoint or --model must be provided")
           
    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Setting for posterity
    args.color_jitter = 0
    args.aa = ""
    args.train_interpolation = "bicubic"
    args.re_prob = 0
    args.remode = ""
    args.recount = 0

    # 確保 DatasetImgTarget 需要的所有參數都存在
    if not hasattr(args, 'folder_val'):
        if hasattr(args, 'folder_test'):
            args.folder_val = args.folder_test
        else:
            args.folder_val = 'images'  # 預設值
    
    if not hasattr(args, 'df_val'):
        if hasattr(args, 'df_test'):
            args.df_val = args.df_test
        else:
            args.df_val = 'val.csv'  # 預設值
    
    # 其他可能需要的參數
    if not hasattr(args, 'train_trainval'):
        args.train_trainval = False

    # 建立資料集
    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 直接創建模型
    _logger.info(f"Creating model directly: {args.model}")
    
    # 創建一個類似模型參數的對象
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.model = args.model
    model_args.input_size = args.input_size
    model_args.keep_rate = args.keep_rate
    model_args.reduction_loc = args.reduction_loc
    
    # 添加其他可能需要的參數
    if hasattr(args, 'ifa_head'):
        model_args.ifa_head = args.ifa_head
    if hasattr(args, 'ifa_dws_conv_groups'):
        model_args.ifa_dws_conv_groups = args.ifa_dws_conv_groups
    if hasattr(args, 'clc'):
        model_args.clc = args.clc
    if hasattr(args, 'num_clr'):
        model_args.num_clr = args.num_clr
    if hasattr(args, 'clc_include_gap'):
        model_args.clc_include_gap = args.clc_include_gap
    if hasattr(args, 'clc_pool_cls'):
        model_args.clc_pool_cls = args.clc_pool_cls
    if hasattr(args, 'clc_recover_at_last'):
        model_args.clc_recover_at_last = args.clc_recover_at_last
        
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        num_classes=args.num_classes,
        img_size=args.input_size,
        args=model_args
    )
    
    # 如果需要添加 CLR
    if hasattr(args, 'num_clr') and args.num_clr > 0 and hasattr(model, 'add_clr'):
        model.add_clr(args.num_clr)

    model.viz_mode = args.viz_mode

    # 處理新的模型類型
    if hasattr(model_args, 'num_clr') and model_args.num_clr > 0:
        model.add_clr(model_args.num_clr)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Loaded checkpoint: ', args.finetune)

    _logger.info("counting parameters")

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("logging")
    _logger.info('Model %s created, param count: %d' % (model_args.model, param_count))

    _logger.info("moving to device")
    model.to(device)
    model.eval()

    _logger.info("Setting up Loss")

    if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    model_name = model_args.model

    if hasattr(model_args, "heuristic_pattern"):
        model_name = model_name + "-" + model_args.heuristic_pattern

    # 创建可视化目录
    viz_dir = create_viz_dir(args)

    model_data_dict = {"Model": model_name,
                       "Ratio": model_args.keep_rate if hasattr(model_args, "keep_rate") else None,
                       "Location": model_args.reduction_loc if hasattr(model_args, "reduction_loc") else None}

    # 處理不同資料集的圖像名稱
    if args.dataset_name and args.dataset_name.lower() == "imagenet":
        image_names = [os.path.basename(s[0]) for s in dataset_val.samples]
    elif args.dataset_name and args.dataset_name.lower() == "nabirds":
        image_names = [dataset_val.data.iloc[idx].img_id for idx in range(len(dataset_val))]
    elif args.dataset_name and args.dataset_name.lower() == "coco":
        image_names = [dataset_val.ids[idx] for idx in range(len(dataset_val))]
    elif args.dataset_name and args.dataset_name.lower() == "nuswide":
        image_names = [os.path.splitext(os.path.basename(x[0]))[0] for x in dataset_val.itemlist]
    elif hasattr(dataset_val, 'samples'):
        image_names = [os.path.basename(s[0]) for s in dataset_val.samples]
    else:
        image_names = [f"image_{i}" for i in range(len(dataset_val))]

    _logger.info("Ready for Inference")

    if args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide":            
        Sig = torch.nn.Sigmoid()
        preds_regular = []
        targets = []

    with torch.no_grad():
        end = time.time()

        img_count = 0
        viz_count = 0  # 用于限制可视化图像数量
        for batch_idx, (input, target) in enumerate(data_loader_val):
            target = target.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)

            # compute output
            with amp_autocast():
                output = model(input)

            if args.viz_mode:
                
                output, viz_data = output
                viz_keys = list(viz_data.keys())

                kept_tokens = True if "Kept_Tokens" in viz_keys else False
                kept_tokens_abs = True if "Kept_Tokens_Abs" in viz_keys else False
                assign_maps = True if "Assignment_Maps" in viz_keys else False
                soft_assign_maps = False #soft_assign_maps = True if "Soft_Assignment_Maps" in viz_keys else False
                center_feats = False # center_feats = True if "Center_Feats" in viz_keys else False
                fusion_assign = False # fusion_assign = True if "Fusion_Assign" in viz_keys else False
                
            if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                loss = criterion(output, target)
            elif args.dataset.lower() == "coco":
                target = target.max(dim=1)[0].float()
                output = output.float()
                loss = criterion(output, target)
            elif args.dataset.lower() == "nuswide":
                loss = criterion(output.float(), target.float())
            
            batch_size = input.shape[0]
            losses.update(loss.item(), input.size(0))
            
            if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                _, pred = output.topk(5, 1, True, True)
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))
            else:
                # Measure mAP
                pred = Sig(output)
                preds_regular.append(pred.cpu().detach())
                targets.append(target.cpu().detach())

            for i in range(target.shape[0]):
                image_name = image_names[img_count + i]

                data_dict = {"Predictions": pred[i].cpu().numpy(),
                             "Target": target[i].cpu().numpy(),
                             "Loss": loss.item()}
                if args.viz_mode and viz_dir and viz_count < args.viz_max_images:
                    img_viz_dir = os.path.join(viz_dir, f"{image_name}")
                    if not os.path.exists(img_viz_dir):
                        os.makedirs(img_viz_dir)
                    
                    orig_img = input[i].cpu().numpy()
                    orig_img = np.transpose(orig_img, (1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    orig_img = std * orig_img + mean
                    orig_img = np.clip(orig_img, 0, 1)
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(orig_img)
                    plt.title(f"Original Image: {image_name}\nPrediction: {pred[i].cpu().numpy().argmax()}, Target: {target[i].cpu().numpy()}")
                    plt.axis('off')
                    plt.savefig(os.path.join(img_viz_dir, f"original.{args.viz_save_format}"), bbox_inches='tight')
                    plt.close()
                    
                    all_stage_token_indices = {}
                    
                    # First loop: collect token indices for all stages
                    for stage_idx, stage in enumerate(model.get_reduction_count()):
                        stage_name = f"Stage-{stage}"
                        
                        if kept_tokens_abs:
                            # Use absolute indices directly
                            all_stage_token_indices[stage_name] = viz_data["Kept_Tokens_Abs"][stage][i]
                        elif stage_idx == 0 and kept_tokens:
                            # Use direct indices for the first stage
                            all_stage_token_indices[stage_name] = viz_data["Kept_Tokens"][stage][i]
                        elif stage_idx > 0 and kept_tokens:
                            # Subsequent stages require mapping to the previous stage
                            rel_idx = viz_data["Kept_Tokens"][stage][i]
                            prev_stage_name = f"Stage-{model.get_reduction_count()[stage_idx-1]}"
                            
                            if not "evit" in model_args.model:
                                rel_idx = rel_idx[rel_idx >= 0]
                            
                            if prev_stage_name in all_stage_token_indices:
                                prev_indices = all_stage_token_indices[prev_stage_name]
                                valid_rel_idx = rel_idx[rel_idx < len(prev_indices)]
                                all_stage_token_indices[stage_name] = prev_indices[valid_rel_idx]
                            else:
                                # If no previous stage info is available, use relative indices
                                _logger.warning(f"Missing previous stage indices for {stage_name}")
                                all_stage_token_indices[stage_name] = rel_idx
                    
                    # Second loop: visualization processing
                    for stage_idx, stage in enumerate(model.get_reduction_count()):
                        stage_name = f"Stage-{stage}"
                        data_dict[stage_name] = {}
                        
                        # Visualize kept tokens
                        if stage_name in all_stage_token_indices:
                            kept_token_idx = all_stage_token_indices[stage_name]
                            data_dict[stage_name]["Kept_Token"] = kept_token_idx
                            
                            # Visualize token positions
                            if args.viz_tokens:
                                token_pos_path = os.path.join(img_viz_dir, f"{stage_name}_tokens.{args.viz_save_format}")
                                try:
                                    visualize_token_positions((args.input_size, args.input_size), 
                                                            kept_token_idx, 
                                                            token_pos_path,
                                                            f"{stage_name} - Token Positions",
                                                            model_type=model_args.model,
                                                            highlight_kept=not args.viz_highlight_dropped)
                                    
                                    # Overlay visualization on the original image
                                    token_overlay_path = os.path.join(img_viz_dir, f"{stage_name}_tokens_overlay.{args.viz_save_format}")
                                    visualize_token_positions_on_image(orig_img, 
                                                                    kept_token_idx, 
                                                                    token_overlay_path,
                                                                    f"{stage_name} - Token Overlay",
                                                                    model_type=model_args.model,
                                                                    highlight_kept=not args.viz_highlight_dropped)
                                except Exception as e:
                                    _logger.error(f"Error visualizing token positions for {stage_name}: {e}")
                        
                        # Visualize attention maps
                        if assign_maps and args.viz_attn:
                            try:
                                assignment_maps = viz_data["Assignment_Maps"][stage][i]
                                data_dict[stage_name]["Assignment_Maps"] = assignment_maps
                                
                                # Average multi-head attention maps into 2D
                                if len(assignment_maps.shape) > 2:
                                    avg_attn_map = np.mean(assignment_maps, axis=0)
                                    
                                    num_heads = min(4, assignment_maps.shape[0])  # Visualize up to 4 attention heads
                                    for head_idx in range(num_heads):
                                        head_attn = assignment_maps[head_idx]
                                        h = int(np.sqrt(head_attn.shape[0]))
                                        w = h
                                        head_attn_2d = head_attn.reshape(h, w)
                                        
                                        head_path = os.path.join(img_viz_dir, f"{stage_name}_attn_head{head_idx}.{args.viz_save_format}")
                                        visualize_attention_map(head_attn_2d, head_path, f"{stage_name} - Attention Head {head_idx}")
                                else:
                                    avg_attn_map = assignment_maps
                                    
                                # Reshape into 2D map
                                h = int(np.sqrt(avg_attn_map.shape[0]))
                                w = h
                                attn_map_2d = avg_attn_map.reshape(h, w)
                                
                                attn_path = os.path.join(img_viz_dir, f"{stage_name}_attn.{args.viz_save_format}")
                                visualize_attention_map(attn_map_2d, attn_path, f"{stage_name} - Attention Map", cmap='hot')

                                attn_overlay_path = os.path.join(img_viz_dir, f"{stage_name}_attn_overlay.{args.viz_save_format}")
                                try:
                                    visualize_attention_on_image(orig_img, attn_map_2d, attn_overlay_path, 
                                                            f"{stage_name} - Attention Overlay")
                                except Exception as e:
                                    _logger.error(f"Error visualizing attention overlay for {stage_name}: {e}")
                            except Exception as e:
                                _logger.error(f"Error visualizing attention maps for {stage_name}: {e}")
                        
                        # Visualize feature maps
                        if center_feats and args.viz_features:
                            try:
                                center_features = viz_data["Center_Feats"][stage][i]
                                data_dict[stage_name]["Center_Feats"] = center_features
                                
                                # Average or select part of the feature channels
                                if len(center_features.shape) > 1:
                                    # For high-dimensional features, only visualize some channels
                                    num_channels = min(5, center_features.shape[0])
                                    for ch_idx in range(num_channels):
                                        feat_map = center_features[ch_idx]
                                        if len(feat_map.shape) == 1:
                                            h = int(np.sqrt(feat_map.shape[0]))
                                            w = h
                                            feat_map = feat_map.reshape(h, w)
                                        
                                        feat_path = os.path.join(img_viz_dir, f"{stage_name}_feat_ch{ch_idx}.{args.viz_save_format}")
                                        visualize_feature_map(feat_map, feat_path, title=f"{stage_name} - Feature Channel {ch_idx}")
                                    
                                    # Also create visualization for the average feature map
                                    avg_feat_map = np.mean(center_features, axis=0)
                                    if len(avg_feat_map.shape) == 1:
                                        h = int(np.sqrt(avg_feat_map.shape[0]))
                                        w = h
                                        avg_feat_map = avg_feat_map.reshape(h, w)
                                    
                                    avg_feat_path = os.path.join(img_viz_dir, f"{stage_name}_feat_avg.{args.viz_save_format}")
                                    visualize_feature_map(avg_feat_map, avg_feat_path, title=f"{stage_name} - Average Feature Map")
                                else:
                                    # If it's a single-channel feature
                                    feat_map = center_features
                                    if len(feat_map.shape) == 1:
                                        h = int(np.sqrt(feat_map.shape[0]))
                                        w = h
                                        feat_map = feat_map.reshape(h, w)
                                    
                                    feat_path = os.path.join(img_viz_dir, f"{stage_name}_feat.{args.viz_save_format}")
                                    visualize_feature_map(feat_map, feat_path, title=f"{stage_name} - Feature Map")
                            except Exception as e:
                                _logger.error(f"Error visualizing features for {stage_name}: {e}")
                    
                    viz_count += 1
                model_data_dict[image_name] = data_dict

            img_count += target.shape[0]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                if args.dataset.lower() != "coco" and args.dataset.lower() != "nuswide":
                    _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                        'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                            batch_idx, len(data_loader_val), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg,
                            loss=losses, top1=top1, top5=top5))
                else:
                        _logger.info(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
                            batch_idx, len(data_loader_val), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg))
    
        
    if args.dataset.lower() == "coco" or args.dataset.lower() == "nuswide":
        mAP_score = utils.mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
        top1.update(mAP_score, 1)
        top5.update(mAP_score, 1)

    top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=args.input_size)

    model_data_dict["Top1-Acc"] = round(top1a, 4)
    model_data_dict["Top5-Acc"] = round(top5a, 4)
    model_data_dict["Params"] = round(param_count / 1e6, 2)

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return model_data_dict

def main(args, _logger):  
    
    viz_data = validate(args, _logger)
    viz_data_file = os.path.join(args.output_dir, args.viz_output_name)
    write_viz(viz_data_file, viz_data)

def write_viz(viz_file, viz_data):
    with open(viz_file, "w") as write_file:
        json.dump(viz_data, write_file, indent=4, cls=NumpyArrayEncoder)

if __name__ == '__main__':

    from timm.utils import setup_default_logging
    import logging

    _logger = logging.getLogger('validate')
    setup_default_logging()
    args = parser.parse_args()
    adjust_config(args)
    main(args, _logger)