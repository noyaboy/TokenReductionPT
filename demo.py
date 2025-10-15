import os
import numpy as np
import torch
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import cv2
import argparse
from timm.models import create_model
import logging
from contextlib import redirect_stdout
import io
import sys

# 從 validate.py 導入必要的函數
from validate import visualize_token_positions_on_image, yaml_config_hook
from datasets import build_dataset

# 設定日誌
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('demo')

# 解析配置文件的函數
def parse_config(config_path):
    if config_path and os.path.exists(config_path):
        config = yaml_config_hook(os.path.abspath(config_path))
        args = argparse.Namespace()
        for k, v in config.items():
            setattr(args, k, v)
        return args
    return argparse.Namespace()

# 修改後的驗證函數，只處理一張圖片
def inference_single(config_path, image_path, model_name, keep_rate, reduction_loc):
    # 首先從配置文件載入基本參數
    args = parse_config(config_path)
    
    # 更新參數
    args.model = model_name
    args.keep_rate = [float(keep_rate)] if isinstance(keep_rate, (str, float)) else keep_rate
    args.reduction_loc = [int(x) for x in reduction_loc.split(',')] if isinstance(reduction_loc, str) else reduction_loc
    args.pretrained = True
    args.viz_mode = True
    args.input_size = getattr(args, 'input_size', 224)
    args.resize_size = None
    args.custom_mean_std = True
    args.train_trainval = True
    args.finetune = None
    
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)
    
    # 載入模型
    model_args = argparse.Namespace()
    model_args.model = args.model
    model_args.input_size = args.input_size
    model_args.keep_rate = args.keep_rate
    model_args.reduction_loc = args.reduction_loc
    
    for param in ['ifa_head', 'ifa_dws_conv_groups', 'clc', 'num_clr', 
                 'clc_include_gap', 'clc_pool_cls', 'clc_recover_at_last']:
        if hasattr(args, param):
            setattr(model_args, param, getattr(args, param))
    
    print(f"載入模型: {args.model}, num_classes: {args.num_classes}")
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        num_classes=args.num_classes,
        img_size=args.input_size,
        args=model_args
    )
    
    # 如果指定了檢查點，載入權重
    if hasattr(args, 'finetune') and args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Loaded checkpoint: ', args.finetune)
    
    model.viz_mode = True
    model.to(device)
    model.eval()
    
    # 處理輸入圖像
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 進行推論
    with torch.no_grad():
        output, viz_data = model(img_tensor)
    
    # 處理預測結果
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # 取得可視化資料
    viz_keys = list(viz_data.keys())
    kept_tokens = "Kept_Tokens" in viz_keys
    kept_tokens_abs = "Kept_Tokens_Abs" in viz_keys
    
    # 儲存可視化結果
    stage_images = {}
    
    # 轉換為 numpy 陣列用於可視化
    orig_img = img_tensor[0].cpu().numpy()
    orig_img = np.transpose(orig_img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = std * orig_img + mean
    orig_img = np.clip(orig_img, 0, 1)
    
    # 為每個階段創建可視化
    all_stage_token_indices = {}
    
    # 收集所有階段的 token 索引
    for stage_idx, stage in enumerate(model.get_reduction_count()):
        stage_name = f"Stage-{stage}"
        
        if kept_tokens_abs:
            all_stage_token_indices[stage_name] = viz_data["Kept_Tokens_Abs"][stage][0]
        elif stage_idx == 0 and kept_tokens:
            all_stage_token_indices[stage_name] = viz_data["Kept_Tokens"][stage][0]
        elif stage_idx > 0 and kept_tokens:
            rel_idx = viz_data["Kept_Tokens"][stage][0]
            prev_stage_name = f"Stage-{model.get_reduction_count()[stage_idx-1]}"
            
            if not "evit" in model_args.model:
                rel_idx = rel_idx[rel_idx >= 0]
            
            if prev_stage_name in all_stage_token_indices:
                prev_indices = all_stage_token_indices[prev_stage_name]
                valid_rel_idx = rel_idx[rel_idx < len(prev_indices)]
                all_stage_token_indices[stage_name] = prev_indices[valid_rel_idx]
            else:
                all_stage_token_indices[stage_name] = rel_idx
    
    # 為每個階段創建可視化
    for stage_idx, stage in enumerate(model.get_reduction_count()):
        stage_name = f"Stage-{stage}"
        
        if stage_name in all_stage_token_indices:
            kept_token_idx = all_stage_token_indices[stage_name]
            
            # 使用直接計算方式來生成可視化
            h, w = orig_img.shape[0], orig_img.shape[1]
            patch_size = 16  # 預設 patch 大小
            
            # 確保 token_positions 是 numpy 陣列
            if not isinstance(kept_token_idx, np.ndarray):
                kept_token_idx = np.array(kept_token_idx)
            
            # 過濾無效的索引值
            valid_indices = kept_token_idx[kept_token_idx >= 0]
            
            num_patches_h = h // patch_size
            num_patches_w = w // patch_size
            total_patches = num_patches_h * num_patches_w
            
            # 創建所有 token 的位置索引
            all_token_indices = np.arange(total_patches)
            
            # 高亮顯示丟棄的 token
            highlight_indices = np.array([idx for idx in all_token_indices if idx not in valid_indices])
            
            # 創建遮罩
            mask = np.zeros((h, w, 4), dtype=np.float32)
            
            # 將丟棄的 token 標記為紅色
            for pos in highlight_indices:
                if pos < total_patches:
                    row = (pos // num_patches_w) * patch_size
                    col = (pos % num_patches_w) * patch_size
                    if row < h and col < w:
                        mask[row:min(row+patch_size, h), col:min(col+patch_size, w), 0] = 1.0  # 紅色
                        mask[row:min(row+patch_size, h), col:min(col+patch_size, w), 3] = 0.4  # 透明度
            
            # 創建可視化圖像
            overlay = orig_img.copy()
            for i in range(h):
                for j in range(w):
                    if mask[i, j, 3] > 0:
                        overlay[i, j, 0] = overlay[i, j, 0] * (1 - mask[i, j, 3]) + mask[i, j, 0] * mask[i, j, 3]
                        overlay[i, j, 1] = overlay[i, j, 1] * (1 - mask[i, j, 3])
                        overlay[i, j, 2] = overlay[i, j, 2] * (1 - mask[i, j, 3])
            
            # 添加網格線
            for i in range(0, h, patch_size):
                overlay[i:i+1, :] = [1.0, 1.0, 1.0]
            for j in range(0, w, patch_size):
                overlay[:, j:j+1] = [1.0, 1.0, 1.0]
            
            stage_images[stage_name] = overlay
    
    # 創建一個完整的可視化圖像，包含所有階段
    stages = list(stage_images.keys())
    num_stages = len(stages)
    
    if num_stages == 0:
        return f"Class: {predicted_class}, Confidence: {confidence:.4f}", None
    
    # 組合所有階段的圖像
    combined_height = orig_img.shape[0]
    combined_width = orig_img.shape[1] * num_stages
    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.float32)
    
    for i, stage in enumerate(stages):
        start_col = i * orig_img.shape[1]
        end_col = start_col + orig_img.shape[1]
        combined_img[:, start_col:end_col] = stage_images[stage]
    
    # 轉換為 uint8 格式
    combined_img = (combined_img * 255).astype(np.uint8)
    
    return f"Class: {predicted_class}, Confidence: {confidence:.4f}", combined_img

# 主函數，用於處理 gradio 介面的輸入
def demo(image_path, model_name='evit_deit_tiny_patch16_224.fb_in1k', keep_rate=0.7, reduction_loc="3,6,9", config_path="configs/cotton_ft_weakaugs.yaml"):
    try:
        # 執行單張圖片的推論
        prediction, combined_img = inference_single(config_path, image_path, model_name, keep_rate, reduction_loc)
        return prediction, combined_img
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"錯誤: {str(e)}", None

# Create Gradio interface
title = 'Token Reduction Visualization Demo'
description = 'Upload an image to see the visualization of token reduction. Tokens that are discarded are highlighted in red.'
article = '''<p style='text-align: center'>
    Token Reduction Visualization Demo
    </p>'''

# Define input components
inputs = [
    gr.components.Image(type='filepath', label='Upload Image'),
    gr.components.Dropdown(
        choices=[
            'evit_vit_base_patch16_224.mae',
            'evit_vit_base_patch16_clip_224.laion2b',
            'evit_deit_tiny_patch16_224.fb_in1k',
            'evit_deit_small_patch16_224.fb_in1k',
            'evit_deit_base_patch16_224.fb_in1k'
        ],
        value='evit_vit_base_patch16_224.mae',
        label='Model'
    ),
    gr.components.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label='Keep Rate'),
    gr.components.Textbox(value='3,6,9', label='Reduction Locations (comma-separated)'),
    gr.components.Textbox(value='configs/cotton_ft_weakaugs.yaml', label='Config File Path')
]

# Define output components
outputs = [
    gr.components.Textbox(label='Prediction Result'),
    gr.components.Image(label='Token Reduction Visualization', type='numpy')
]

# Prepare examples
examples = [
    ['../../data/cotton/cotton_square_new/1_1_train.png'],
    ['../../data/cotton/cotton_square_new/1_2_test.png']
]

# Create and launch the Gradio interface
gr.Interface(
    demo, inputs, outputs,
    title=title,
    description=description,
    article=article,
    examples=examples if os.path.exists('../../data/cotton/cotton_square_new/1_1_train.png') else None
).launch(debug=True, share=True)