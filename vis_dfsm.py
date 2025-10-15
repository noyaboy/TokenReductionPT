import os
import argparse
from typing import List
from functools import partial
from contextlib import suppress

import cv2
import wandb
import matplotlib.pyplot as plt
from einops import reduce, rearrange
# from timm.models import create_model
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from torchcam import methods

# import models
# import utils
from datasets import build_dataset
from train import get_args_parser, adjust_config, set_seed, set_run_name, build_model
from utils import inverse_normalize


class VisHook:
    def __init__(self,
                 model: nn.Module,
                 model_name: str = 'model1',
                 model_layers: List[str] = None,
                 vis_mask: str = None,
                 device: str ='cpu',
                 debugging: bool = True):
        """
        :param model: (nn.Module) Neural Network 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.model_name = model_name
        self.device = device
        self.debugging = debugging

        self.model_info = {}

        self.model_info['Layers'] = []

        self.model_features = {}

        self.model_layers = model_layers

        self.vis_mask = vis_mask

        if vis_mask and 'CAM' in vis_mask:
            # for _, param in self.model.named_parameters():
            #    param.requires_grad = True

            self.model = self.model.to(self.device)

            for name, layer in self.model.named_modules():
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]

            self.extractor = methods.__dict__[vis_mask](self.model, target_layer=self.model_info['Layers'][-1], enable_hooks=True)

        else:
            self._insert_hooks()
            self.model = self.model.to(self.device)

            self.model.eval()

        print(self.model_info)


    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")


    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))


    def inference(self, images) -> None:
        """
        Computes the attention rollout for the image(s)
        :param x: (input tensor)
        """
        self.model_features = {}

        images = images.to(self.device)

        if self.vis_mask and 'CAM' in self.vis_mask:
            masks = []
            for i, image in enumerate(images):
                self.extractor._hooks_enabled = True
                self.model.zero_grad()

                preds = self.model(image.unsqueeze(0))

                # cal and other models that return crops and aux predictions
                if isinstance(preds, tuple):
                    preds = preds[0]

                _, class_idx = torch.max(preds, dim=-1)
                activation_map = self.extractor(class_idx.item(), preds[i:i+1])[0]

                self.extractor.remove_hooks()
                self.extractor._hooks_enabled = False

                masks.append(activation_map)

            return preds, masks

        preds = self.model(images)

        # cal and other models that return crops and aux predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        return preds, self.model_features


    def inference_save_vis(self, images, fp, pred_text=False, custom_mean_std=False,
                           vis_mask_pow=False, vis_cols=None, reg_reduction='mean'):
        preds, features = self.inference(images)
        preds_top1 = torch.topk(preds, k=1).indices.squeeze()

        regs = 4 if 'reg4' in self.model_name else 0
        masks = calc_masks(features, self.vis_mask, regs=regs)

        masked_imgs = []
        for i in range(images.shape[0]):
            img_unnorm = inverse_normalize(images[i].detach().clone(), custom_mean_std)
            img_masked = apply_mask(img_unnorm, masks[i], vis_mask_pow)
            masked_imgs.append(img_masked)

        number_imgs = len(masked_imgs)
        ncols = vis_cols if vis_cols else int(number_imgs ** 0.5)
        nrows = number_imgs // ncols
        number_imgs = ncols * nrows

        fig = plt.figure(figsize=(ncols, nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                        axes_pad=(0.01, 0.01), direction='row', aspect=True)

        for i, (ax, np_arr) in enumerate(zip(grid, masked_imgs)):
            ax.axis('off')
            ax.imshow(np_arr)
            if pred_text:
                text = preds_top1[i].item()
                ax.text(0.5, -0.01, text, transform=ax.transAxes,
                        ha='center', va='top', fontsize=8)

        save_images(fig, fp, self.debugging)

        return preds, masked_imgs[0]


def save_images(fig, fp, debugging=True):
    fig.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print('Saved ', fp)

    if not debugging:
        save_name = os.path.splitext(os.path.split(fp)[-1])[0]
        wandb.log({f'{save_name}': wandb.Image(fig)})

    return 0


def hyperbolic_distance(t1: torch.Tensor, t2: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Lorentzian Distance Learning for Hyperbolic Representations
    https://proceedings.mlr.press/v97/law19a.html
    Compute the hyperbolic distance (Poincaré model) between t1 (B, 1, D) and t2 (B, S, D).
    Assumes points are inside the Poincaré ball with curvature c.
    """
    t1_norm = torch.norm(t1, p=2, dim=-1)
    t2_norm = torch.norm(t2, p=2, dim=-1)

    # num = torch.norm(t1 - t2, p=2, dim=-1) ** 2
    num = torch.sum((t1 - t2) ** 2, dim=-1)
    denom = (1 - c * t1_norm ** 2) * (1 - c * t2_norm ** 2)

    # Avoid division by zero or invalid values
    safe_denom = torch.clamp(denom, min=1e-10)
    argument = 1 + 2 * c * num / safe_denom
    argument = torch.clamp(argument, min=1.0)  # Ensure valid input for acosh

    return (1 / (c ** 0.5)) * torch.acosh(argument)


def lorentzian_inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Lorentzian inner product for two embeddings x and y in the hyperboloid model.
    x and y should be of shape (B, S+1), where the last dimension includes the time component.
    """
    space_prod = torch.sum(x[..., :-1] * y[..., :-1], dim=-1)  # Dot product over spatial dimensions
    time_prod = x[..., -1] * y[..., -1]  # Time component product
    return space_prod - time_prod  # Lorentzian inner product


def exponential_map(u: torch.Tensor, c: float) -> torch.Tensor:
    """
    Apply the exponential map at the origin to lift embeddings to the hyperboloid.
    Args:
        u: Tensor of shape (..., D), input embeddings in Euclidean space.
        c: Curvature of the hyperbolic space.
    Returns:
        Tensor of shape (..., D+1) lifted to the hyperboloid.
    """
    u = ((1 / u.shape[-1]) ** 0.5) * u
    u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
    # u_norm = torch.norm(u, p=2, dim=-1)

    num = torch.sinh((c ** 0.5) * u_norm)
    denom = ((c ** 0.5) * u_norm + 1e-9)  # Avoid division by zero
    scale_factor = num / denom

    x_space = scale_factor * u

    x_space_norm = torch.norm(x_space, p=2, dim=-1, keepdim=True)

    # x_time = torch.cosh((c ** 0.5) * u_norm)
    x_time = torch.sqrt((c ** 0.5) + x_space_norm ** 2)

    # return torch.cat([scale_factor * u, time_component], dim=-1)  # Shape (..., D+1)
    return torch.cat([x_space, x_time], dim=-1)  # Shape (..., D+1)


def hyperbolic_distance_lorentz(t1: torch.Tensor, t2: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Embedding Geometries of Contrastive Language-Image Pre-Training
    https://www.arxiv.org/abs/2409.13079
    Hyperbolic Image-Text Representations
    https://proceedings.mlr.press/v202/desai23a/desai23a.pdf
    Compute the hyperbolic distance based on the Lorentzian distance on the hyperboloid model.
    Args:
        t1: Tensor of shape (B, 1, D), input embeddings.
        t2: Tensor of shape (B, S, D), target embeddings.
        c: Curvature of the hyperbolic space (default 1.0).
    Returns:
        Tensor of shape (B, S) representing the hyperbolic distances.
    """
    # Lift to hyperboloid using the exponential map
    x_hyperboloid = exponential_map(t1, c)  # Shape (B, 1, D+1)
    y_hyperboloid = exponential_map(t2, c)  # Shape (B, S, D+1)

    # Compute Lorentzian inner product
    lorentzian_ip = lorentzian_inner_product(x_hyperboloid, y_hyperboloid)  # Shape (B, S)

    # Hyperbolic distance
    distance = ((1 / c) ** 0.5) * torch.acosh(-lorentzian_ip)

    # Clamp values to prevent numerical issues with acosh
    # lorentzian_ip = torch.clamp(lorentzian_ip, max=-1e-5)

    # Hyperbolic distance
    # distance = ((1 / c) ** 0.5) * torch.acosh(torch.clamp(-lorentzian_ip, min=1.0))
    return distance


def lorentzian_distance(u: torch.Tensor, v: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Lorentzian Distance Learning for Hyperbolic Representations
    # eq 2 and 6
    # the sum is over both u_time and u not only over u (as in eq 2)
    https://proceedings.mlr.press/v97/law19a.html
    https://github.com/MarcTLaw/LorentzianDistanceRetrieval/blob/master/lorentzian_model.py
    Compute the Lorentzian distance between u and v.
    Args:
        u: Tensor of shape (B, 1, D).
        v: Tensor of shape (B, S, D).
    Returns:
        Tensor of shape (B, S) representing the Lorentzian distances.
    """
    u_0 = torch.sqrt(torch.sum(u ** 2, dim=-1, keepdim=True) + beta)
    # why is this negative? typo? if this is negative and we use the 
    # commented equation with the torch.sum() the result is equal
    # probably chatgpt/the authors exploit linear algebra properties
    # when writing these equations, i follow the originals for simplicity
    # v_0 = -torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + beta)
    v_0 = torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + beta)

    u_lorentz = torch.cat([u, u_0], dim=-1)  # Shape (B, D+1)
    v_lorentz = torch.cat([v, v_0], dim=-1)  # Shape (B, D+1)

    # Compute Lorentzian inner product
    lorentzian_ip = lorentzian_inner_product(u_lorentz, v_lorentz)  # Shape (B, S)
    result = -2 * beta - 2 * lorentzian_ip  # Shape (B, S)
    # result = -2 * beta - 2 * torch.sum(u_lorentz * v_lorentz, dim=-1)
    return result
    

def min_max_normalize(tensor):
    # Find the min and max values in the tensor
    min_val = tensor.min()
    max_val = tensor.max()

    # Apply min-max normalization
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor


def calc_masks(features, vis_mask='rollout', def_rollout_end=4, regs=0):
    if vis_mask is None:
        bs = list(features.values())[0].shape[0]
        masks = [None for _ in range(bs)]
        return masks


    elif 'rollout' in vis_mask or 'attention' in vis_mask:
        features = {k: v for k, v in features.items() if 'attn' in k}
        print(features.keys())
        features = list(features.values())


    elif 'maws' in vis_mask:
        assert regs == 0, 'MAWS does not work with Registers'
        features = {k: v for k, v in features.items() if 'attn' in k}
        print(features.keys())


    elif 'gls' in vis_mask or 'feats' in vis_mask:
        features = {k: v for k, v in features.items() if 'norm' in k}
        print(features.keys())
        features = list(features.values())


    if 'rollout' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            rollout_start = 0
            rollout_end = int(def_rollout_end)
        elif len(splits) == 2:
            rollout_start = 0
            rollout_end = int(splits[-1])
        elif len(splits) == 3:
            rollout_start = int(splits[1])
            rollout_end = int(splits[-1])
        else:
            raise NotImplementedError

        # input: list of length L with each element shape: B, NH, S, S
        # use only first 4 (should be similar enough to full rollout)
        # because scores after 4 should have different shape
        # output: B, S, S
        # select 1st token attention for the rest []:, 0, 1:] -> B, S-1

        attention = features[rollout_start:rollout_end]
        attention = attention_rollout(attention)
        if regs > 0:
            masks = attention[:, 0, 1+regs:]
        else:
            masks = attention[:, 0, 1:]


    elif 'dmaws' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = 1
        elif len(splits) == 2:
            layer = int(splits[-1])
        else:
            raise NotImplementedError

        masks = calc_dmaws(features, layer)

    elif 'maws' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = 0
        elif len(splits) == 2:
            layer = int(splits[-1])
        else:
            raise NotImplementedError

        attentions = [
            features[f'blocks.{layer}.attn.pre_softmax'],
            features[f'blocks.{layer}.attn.attn_drop'],
        ]

        masks = calc_maws(attentions)

    elif 'attention' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = 0
        elif len(splits) == 2:
            layer = int(splits[-1])
        else:
            raise NotImplementedError

        attention = features[layer]
        attention = reduce(attention, 'b h s1 s2 -> b s1 s2', 'mean')
        if regs > 0:
            masks = attention[:, 0, 1+regs:]
        else:
            masks = attention[:, 0, 1:]


    elif 'gls' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = -1
        elif len(splits) == 2:
            layer = int(splits[-1])
        else:
            raise NotImplementedError

        features = features[layer]

        fh = int(features.shape[1] ** 0.5)
        if fh ** 2 == features.shape[1]:
            g = reduce(features, 'b s d -> b 1 d', 'mean')
            l = features
        else:
            g = features[:, :1]
            l = features[:, 1:]

        if 'glsl2' in vis_mask:
            # l2 distance
            masks = torch.norm(g - l, p=2, dim=-1)
            masks = min_max_normalize(masks)
        elif 'glslz' in vis_mask:
            # lorentzian distance
            masks = lorentzian_distance(g, l)
            masks = min_max_normalize(masks)
        elif 'glshblz' in vis_mask:
            # hyperbolic distance lorentz
            masks = hyperbolic_distance_lorentz(g, l)
            masks = min_max_normalize(masks)
        else:
            # cosine similarity
            masks = F.cosine_similarity(g, l, dim=-1)

        # inverted (0 becomes 1 and 1 becomes 0)
        if any([substr in vis_mask for substr in ['igls', 'glsl2', 'glslz', 'glshblz']]):
            masks = -masks + 1


    elif 'feats' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = -1
            channel = 0
        elif len(splits) == 2:
            layer = int(splits[-1])
            channel = 0
        elif len(splits) == 3:
            layer = int(splits[1])
            channel = int(splits[-1])
        else:
            raise NotImplementedError

        features = features[layer]

        masks = features[:, 1:, channel:channel+1]


    elif 'bap' in vis_mask:
        print(features.keys())
        features = list(features.values())[0]

        if vis_mask == 'bap_avg':
            masks = reduce(features, 'b ac ah aw -> b (ah aw)', 'mean')
        else:
            splits = vis_mask.split('_')
            if len(splits) == 1:
                map = 0
            elif len(splits) == 2:
                map = int(splits[-1])
            else:
                raise NotImplementedError

            masks = reduce(features[:, map], 'b ah aw -> b (ah aw)', 'mean')


    elif 'CAM' in vis_mask:
        masks = torch.cat(features, dim=0)
        masks = rearrange(masks, 'b ah aw -> b (ah aw)')

    return masks


def apply_mask(img, mask, power=False, color=True):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''

    if mask is None:
        img = rearrange(img.cpu().detach().numpy(), 'c h w -> h w c')
        img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result)
        return result


    elif len(mask.shape) == 2:
        result = rearrange(mask.cpu().detach().numpy(), '(h w) 1 -> h w', h=int(mask.shape[0] ** 0.5))

        result = result * 255
        result = result.astype('uint8')

        result = cv2.resize(result, (img.shape[-1], img.shape[-1]))
        result = Image.fromarray(result, 'L')
        return result


    if power:
        mask = (mask ** 4)

    # convert to numpy array
    mask = rearrange(mask, '(h w) -> h w', h=int(mask.shape[0] ** 0.5))
    mask = mask.cpu().detach().numpy()

    if mask.dtype == 'float16':
        mask = mask.astype('float32')

    if color:
        mask = cv2.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().detach().numpy(), 'c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    if color:
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        mask = rearrange(mask, 'h w -> h w 1')        

    if color:
        result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    else:
        result = (mask * img)
        result = cv2.normalize(
            result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = result.astype('uint8')

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def attention_rollout(scores_soft):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = torch.stack(scores_soft)

    # Average the attention weights across all heads.
    att_mat = reduce(att_mat, 'l b h s1 s2 -> l b s1 s2', 'mean')

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(), device=att_mat.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    return joint_attentions[-1]


def calc_maws(attns, reduction='mean'):
    # first row of pre_sm_attn * first col of post_sm_attn
    pre_sm_attn, post_sm_attn = attns
    post_sm_attn = reduce(post_sm_attn, 'b l s1 s2 -> b s1 s2', reduction)[:, 0, 1:]
    pre_sm_attn = reduce(pre_sm_attn, 'b l s1 s2 -> b s1 s2', reduction)
    pre_sm_attn = pre_sm_attn.softmax(dim=-2)[:, 1:, 0]
    maws = pre_sm_attn * post_sm_attn
    return maws


def calc_dmaws(attns, l):
    # print(attns.keys(), l, method)
    maws1 = [
        attns[f'blocks.{l - 1}.attn.pre_softmax'],
        attns[f'blocks.{l - 1}.attn.attn_drop'],
    ]
    maws2 = [
        attns[f'blocks.{l}.attn.pre_softmax'],
        attns[f'blocks.{l}.attn.attn_drop'],
    ]
    maws1 = calc_maws(maws1)
    maws2 = calc_maws(maws2)
    cls_attn = maws1 * maws2

    return cls_attn


def get_layers(model, model_name, selector=None, vis_mask=None):
    if vis_mask is None:
        layers = []
        for name, _ in model.named_modules():
            layers.append(name)
        # layers = layers[-2:-1]
        layers = layers[-1]

    elif (any([kw in vis_mask for kw in ['attention', 'rollout', 'gls', 'feats', 'maws']]) and
        not any([kw in model_name for kw in ['vit', 'deit']])):
        raise NotImplementedError

    elif any([kw in model_name for kw in ['vit', 'deit']]) and any(
        [kw in vis_mask for kw in ['CAM']]):
        raise NotImplementedError

    elif any([kw in model_name for kw in ['vit', 'deit']]) and any(
        [kw in vis_mask for kw in ['attention', 'rollout', 'gls', 'feats', 'maws']]):
        # only works for deit/vit base
        layers = []
        for name, _ in model.named_modules():
            # print(name)
            if (any([kw in name for kw in ['attn.drop', 'attn_drop', 'encoder_norm', 'model.norm']])):
                layers.append(name)
            elif vis_mask and (any(kw in vis_mask for kw in ['gls', 'feats'])) and 'norm2' in name:
                layers.append(name)
            elif vis_mask and 'maws' in vis_mask and 'pre_softmax' in name: 
                layers.append(name)

    elif 'bap' in vis_mask and selector == 'cal':
        layers = ['model.dfsm.attentions.bn']

    elif 'bap' in vis_mask and selector != 'cal':
        raise NotImplementedError

    else:

        if model_name == 'vgg19_bn':
            layers = ['model.features.50']
        elif model_name == 'resnet18':
            layers = ['model.layer4.1.bn2']
        elif model_name == 'resnet34':
            layers = ['model.layer4.2.bn2']
        elif 'resnet50' in model_name:
            layers = ['model.layer4.2.bn3']
        elif model_name == 'resnet101':
            layers = ['model.layer4.2.bn3']
        elif 'vit_b' in model_name:
            layers = ['model.encoder.blocks.11.norm2']
        elif 'beitv2_base_patch16_224_in22k' in model_name or 'deit3_base_patch16_224' in model_name:
            layers = ['model.blocks.11.norm2']
        elif 'deit3_large_patch16_224' in model_name:
            layers = ['model.blocks.23.norm2']
        elif model_name == 'van_b3':
            layers = ['model.norm4']
        elif 'convnext' in model_name:
            layers = ['model.stages.3.blocks.2.norm']
        elif 'convnext_base' in model_name:
            layers = ['model.stages.3.blocks.2.norm']
        elif 'swin' in model_name:
            layers = ['model.layers.3.blocks.1.norm2']
        elif 'swin_base' in model_name:
            layers = ['model.layers.3.blocks.1.norm2']
        elif 'resnetv2_101' in model_name:
            layers = ['model.stages.3.blocks.2.norm3']
        else:
            layers = []
            for name, _ in model.named_modules():
                # print(name)
                if 'norm' in name or 'bn' in name:
                    layers.append(name)
            layers = layers[-1]

        if selector == 'cal' and any([kw in model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
            layers = [layer.replace('model.', 'model.encoder.0.') for layer in layers]
        elif selector == 'cal':
            layers = [layer.replace('model.', 'model.encoder.') for layer in layers]

    return layers


def build_environment_inference(args):
    set_seed(args.seed)

    # adjust input sizes
    args.resize_size = args.resize_size if args.resize_size else int(args.input_size / 0.875)
    args.test_input_size = args.test_input_size if args.test_input_size else args.input_size
    args.test_resize_size = args.test_resize_size if args.test_resize_size else int(args.test_input_size / 0.875)


    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=args.shuffle_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_val, shuffle=args.shuffle_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(args.dataset_name, args.num_classes, len(dataset_train), len(dataset_val))


    if args.finetune:
        args.setting = 'ft'
    elif args.pretrained:
        args.setting = 'fz'
    else:
        args.setting = 'scratch'

    set_run_name(args, setting=True)

    if not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name


    model = build_model(args)

    model.to(args.device)
    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=True)
        print('Loaded checkpoint: ', args.finetune)


    layers = get_layers(model, args.model, getattr(args, 'selector', None), args.vis_mask)

    hook = VisHook(model, args.model, layers, args.vis_mask, args.device, args.debugging)

    amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress

    return train_loader, test_loader, hook, amp_autocast


def make_visualizations(args, train_loader, test_loader, hook, amp_autocast):
    pow = '_power' if args.vis_mask_pow else ''

    with amp_autocast():
        if args.vis_wrong_only or args.vis_class:
            wrong_images = []
            wrong_targets = []
            if args.vis_class is not None:
                fp = os.path.join(args.output_dir, f'test_wrong_{args.vis_class}_{args.vis_mask}{pow}.png')
            else:
                fp = os.path.join(args.output_dir, f'test_wrong_{args.vis_mask}{pow}.png')
                
            for i, (images, targets) in enumerate(test_loader):
                targets = targets.to(args.device)

                # break out of loop if reach number of batch size images
                if len(wrong_images) == args.batch_size:
                    break
                # jump to next set of images
                elif args.vis_class and args.vis_class not in targets:
                    continue

                preds = hook.inference(images)[0]
                preds = torch.topk(preds, k=1).indices.squeeze()

                for j, (target, pred) in enumerate(zip(targets, preds)):
                    if len(wrong_images) == args.batch_size:
                        break

                    if target != pred and (not args.vis_class or args.vis_class == target):
                        # print('Target and predictions (vis_class): ', target, pred, args.vis_class)
                        wrong_images.append(images[j])
                        wrong_targets.append(targets[j])

            wrong_images = torch.stack(wrong_images, dim=0)
            wrong_targets = torch.stack(wrong_targets, dim=0)
            args.vis_cols = len(wrong_images)
            preds, _ = hook.inference_save_vis(
                wrong_images, fp, args.vis_pred_text, args.custom_mean_std,
                args.vis_mask_pow, args.vis_cols)
            preds = torch.topk(preds, k=1).indices.squeeze()
            print('Test labels: ', wrong_targets, '\nPreds: ', preds)
            return 0

        else:
            images, targets = next(iter(train_loader))
            fp = os.path.join(args.output_dir, f'train_{args.vis_mask}{pow}.png')

            preds, _ = hook.inference_save_vis(
                images, fp, args.vis_pred_text, args.custom_mean_std,
                args.vis_mask_pow, args.vis_cols)
            preds = torch.topk(preds, k=1).indices.squeeze()
            print('Train labels: ', targets, '\nPreds: ', preds)

            images, targets = next(iter(test_loader))
            fp = os.path.join(args.output_dir, f'test_{args.vis_mask}{pow}.png')

            preds, _ = hook.inference_save_vis(
                images, fp, args.vis_pred_text, args.custom_mean_std,
                args.vis_mask_pow, args.vis_cols)
            preds = torch.topk(preds, k=1).indices.squeeze()
            print('Test labels: ', targets, '\nPreds: ', preds)

    return 0


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--shuffle_train', action='store_false', help='def uses random sampler')
    parser.add_argument('--shuffle_test', action='store_true', help='def uses random sampler')
    parser.add_argument('--vis_mask', type=str, default=None, help='')
    parser.add_argument('--vis_mask_pow', action='store_true',
                        help='power (of 4) masks when applying heatmap')
    parser.add_argument('--vis_class', type=int, default=None)
    parser.add_argument('--vis_wrong_only', action='store_true')
    parser.add_argument('--vis_pred_text', action='store_true')
    parser.add_argument('--vis_cols', type=int, default=None,
                        help='how many columns when visualizing images')
    parser.set_defaults(output_dir='results_inference', batch_size=6)
    args = parser.parse_args()
    adjust_config(args)

    train_loader, test_loader, hook, amp_autocast = build_environment_inference(args)
    set_seed(args.seed)

    make_visualizations(args, train_loader, test_loader, hook, amp_autocast)

    if not args.debugging:
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()

