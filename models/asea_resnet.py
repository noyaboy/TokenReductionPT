# https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/models/resnet.py

"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'asea_resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
import random
from functools import partial
from typing import Any, Set, Dict, List, Optional, Tuple, Type, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations
from einops.layers.torch import Reduce, Rearrange
from einops import repeat, rearrange, reduce

from .model_utils import build_model_with_cfg
from .asea import AxialSqueezeExcitationAttention, LayerScale2D, select_tokens_fuse_pruned


__all__ = ['AseaResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,

            keep_rate: float = 1.0,
            axis: str = '-1',

            axis_squeeze: str = 'mean',
            squeeze_proj: bool = True,
            axis_conv: int = 3,
            axis_norm: bool = True,
            dw_channels: int = 1,
            excitation_proj: bool = True,
            proj_v: bool = True,
            axis_size: int = None,            

            asea_pos: str = 'post_block',
            dropped_axis_fusion: bool = True,

            keep_rate_var: List = [],
            red_train_only: bool = False,
            red_random_perturb: float = 0.0,
            asea_drop_path: float = 0.0,
            asea_ls_init_values: float = 1e-5,

            debugging: bool = False,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        # asea for axial reduction
        self.keep_rate = keep_rate
        assert 0 <= keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

        if (0 < self.keep_rate <= 1):
            self.axis = axis
            self.asea_pos = asea_pos
            self.dropped_axis_fusion = dropped_axis_fusion

            self.asea = AxialSqueezeExcitationAttention(
                dim=outplanes,
                axis=axis,
                axis_squeeze=axis_squeeze,
                squeeze_proj=squeeze_proj,
                axis_conv=axis_conv,
                axis_norm=axis_norm,
                dw_channels=dw_channels,
                excitation_proj=excitation_proj,
                proj_v=proj_v,

                axis_size=axis_squeeze,            
            )

            self.kr_var = keep_rate_var
            if self.kr_var:
                assert len(self.kr_var) == 2, 'keep_rate_var has to be list of len 2'

            self.red_train_only = red_train_only
            self.red_random_perturb = red_random_perturb

            self.asea_ls = LayerScale2D(outplanes, asea_ls_init_values) if asea_ls_init_values else nn.Identity()
            self.asea_drop_path = DropPath(asea_drop_path) if asea_drop_path > 0. else nn.Identity()

        self.debugging = debugging

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'asea') and (0 < self.keep_rate <= 1) and self.asea_pos in ['pre_block']:
            # axis reduction
            # use layerscale to adjust the output of this branch to be close to 0
            # as training starts so it does not affect training dynamics too much
            h, attn = self.asea(x)

            if self.red_random_perturb and self.training:
                # [B, 1, 1, F_W] or [B, 1, F_H, 1]
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    perturb = torch.normal(0, self.red_random_perturb, attn.shape, device=attn.device)
                    attn = attn + perturb

            x = x + self.asea_drop_path(self.asea_ls(h))

            if self.kr_var and self.training:
                kr = random.uniform(self.keep_rate - self.kr_var[0], self.keep_rate + self.kr_var[1])
            else:
                kr = self.keep_rate

            if (self.red_train_only and self.training) or not self.red_train_only:
                x, idx, compl = select_tokens_fuse_pruned(
                    x, attn, self.axis, kr,
                    self.dropped_axis_fusion, self.debugging)

        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut

        x = self.act2(x)

        if hasattr(self, 'asea') and (0 < self.keep_rate <= 1) and self.asea_pos in [
            'in_block', 'post_block', 'post_blockplus', 'parallel', 'parallelplus']:
            # axis reduction
            # use layerscale to adjust the output of this branch to be close to 0
            # as training starts so it does not affect training dynamics too much
            if self.asea_pos in ['in_block', 'post_block', 'post_blockplus']:
                h, attn = self.asea(x)
            elif self.asea_pos in ['parallel', 'parallelplus']:
                h, attn = self.asea(shortcut)

            if self.red_random_perturb and self.training:
                # [B, 1, 1, F_W] or [B, 1, F_H, 1]
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    perturb = torch.normal(0, self.red_random_perturb, attn.shape, device=attn.device)
                    attn = attn + perturb

            # og: shortcut is passed all the way from beginning to end
            if self.asea_pos == 'in_block':
                x = shortcut + self.asea_drop_path(self.asea_ls(h))
            # new: an alternative is to create a shortcut immediately after act2
            elif self.asea_pos == 'post_block':
                x = x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'post_blockplus':
                x = shortcut + x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'parallel':
                x = x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'parallelplus':
                x = shortcut + x + self.asea_drop_path(self.asea_ls(h))

            if self.kr_var and self.training:
                kr = random.uniform(self.keep_rate - self.kr_var[0], self.keep_rate + self.kr_var[1])
            else:
                kr = self.keep_rate

            if (self.red_train_only and self.training) or not self.red_train_only:
                x, idx, compl = select_tokens_fuse_pruned(
                    x, attn, self.axis, kr,
                    self.dropped_axis_fusion, self.debugging)

        # return x, attn, idx, compl
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,

            keep_rate: float = 1.0,
            axis: str = '-1',

            axis_squeeze: str = 'mean',
            squeeze_proj: bool = True,
            axis_conv: int = 3,
            axis_norm: bool = True,
            dw_channels: int = 1,
            excitation_proj: bool = True,
            proj_v: bool = True,
            axis_size: int = None,            

            asea_pos: str = 'post_block',
            dropped_axis_fusion: bool = True,

            keep_rate_var: List = [],
            red_train_only: bool = False,
            red_random_perturb: float = 0.0,
            asea_drop_path: float = 0.0,
            asea_ls_init_values: float = 1e-5,

            debugging: bool = False,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

        # asea for axial reduction
        self.keep_rate = keep_rate
        assert 0 <= keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

        if (0 < self.keep_rate <= 1):
            self.axis = axis
            self.asea_pos = asea_pos
            self.dropped_axis_fusion = dropped_axis_fusion

            self.asea = AxialSqueezeExcitationAttention(
                dim=outplanes,
                axis=axis,
                axis_squeeze=axis_squeeze,
                squeeze_proj=squeeze_proj,
                axis_conv=axis_conv,
                axis_norm=axis_norm,
                dw_channels=dw_channels,
                excitation_proj=excitation_proj,
                proj_v=proj_v,

                axis_size=axis_squeeze,            
            )

            self.kr_var = keep_rate_var
            if self.kr_var:
                assert len(self.kr_var) == 2, 'keep_rate_var has to be list of len 2'

            self.red_train_only = red_train_only
            self.red_random_perturb = red_random_perturb

            self.asea_ls = LayerScale2D(outplanes, asea_ls_init_values) if asea_ls_init_values else nn.Identity()
            self.asea_drop_path = DropPath(asea_drop_path) if asea_drop_path > 0. else nn.Identity()

        self.debugging = debugging

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'asea') and (0 < self.keep_rate <= 1) and self.asea_pos in ['pre_block']:
            # axis reduction
            # use layerscale to adjust the output of this branch to be close to 0
            # as training starts so it does not affect training dynamics too much
            h, attn = self.asea(x)

            if self.red_random_perturb and self.training:
                # [B, 1, 1, F_W] or [B, 1, F_H, 1]
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    perturb = torch.normal(0, self.red_random_perturb, attn.shape, device=attn.device)
                    attn = attn + perturb

            x = x + self.asea_drop_path(self.asea_ls(h))

            if self.kr_var and self.training:
                kr = random.uniform(self.keep_rate - self.kr_var[0], self.keep_rate + self.kr_var[1])
            else:
                kr = self.keep_rate

            if (self.red_train_only and self.training) or not self.red_train_only:
                x, idx, compl = select_tokens_fuse_pruned(
                    x, attn, self.axis, kr,
                    self.dropped_axis_fusion, self.debugging)

        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        if hasattr(self, 'asea') and (0 < self.keep_rate <= 1) and self.asea_pos in [
            'in_block', 'post_block', 'post_blockplus', 'parallel', 'parallelplus']:
            # axis reduction
            # use layerscale to adjust the output of this branch to be close to 0
            # as training starts so it does not affect training dynamics too much
            if self.asea_pos in ['in_block', 'post_block', 'post_blockplus']:
                h, attn = self.asea(x)
            elif self.asea_pos in ['parallel', 'parallelplus']:
                h, attn = self.asea(shortcut)

            if self.red_random_perturb and self.training:
                # [B, 1, 1, F_W] or [B, 1, F_H, 1]
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    perturb = torch.normal(0, self.red_random_perturb, attn.shape, device=attn.device)
                    attn = attn + perturb

            # og: shortcut is passed all the way from beginning to end
            if self.asea_pos == 'in_block':
                x = shortcut + self.asea_drop_path(self.asea_ls(h))
            # new: an alternative is to create a shortcut immediately after act2
            elif self.asea_pos == 'post_block':
                x = x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'post_blockplus':
                x = shortcut + x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'parallel':
                x = x + self.asea_drop_path(self.asea_ls(h))
            elif self.asea_pos == 'parallelplus':
                x = shortcut + x + self.asea_drop_path(self.asea_ls(h))

            if self.kr_var and self.training:
                kr = random.uniform(self.keep_rate - self.kr_var[0], self.keep_rate + self.kr_var[1])
            else:
                kr = self.keep_rate

            if (self.red_train_only and self.training) or not self.red_train_only:
                x, idx, compl = select_tokens_fuse_pruned(
                    x, attn, self.axis, kr,
                    self.dropped_axis_fusion, self.debugging)

        # return x, attn, idx, compl

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob: float = 0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],

        keep_rates: List[float],
        red_axises: List[str],

        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,

        axis_squeeze: str = 'mean',
        squeeze_proj: bool = True,
        axis_conv: int = 3,
        axis_norm: bool = True,
        dw_channels: int = 1,
        excitation_proj: bool = True,
        proj_v: bool = True,
        axis_size: int = None,            

        asea_pos: str = 'post_block',
        dropped_axis_fusion: bool = True,

        keep_rate_var: List = [],
        red_train_only: bool = False,
        red_random_perturb: float = 0.0,
        asea_drop_path: float = 0.0,
        asea_ls_init_values: float = 1e-5,

        debugging: bool = False,

        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, krs, red_axis, db) in enumerate(
        zip(channels, block_repeats, keep_rates, red_axises, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx, kr, red_ax in zip(range(num_blocks), krs, red_axis):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,

                keep_rate=kr,
                axis=red_ax,

                axis_squeeze=axis_squeeze,
                squeeze_proj=squeeze_proj,
                axis_conv=axis_conv,
                axis_norm=axis_norm,
                dw_channels=dw_channels,
                excitation_proj=excitation_proj,
                proj_v=proj_v,

                asea_pos=asea_pos,
                dropped_axis_fusion=dropped_axis_fusion,

                keep_rate_var=keep_rate_var,
                red_train_only=red_train_only,
                red_random_perturb=red_random_perturb,
                asea_drop_path=asea_drop_path,
                asea_ls_init_values=asea_ls_init_values,

                debugging=debugging,

                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class AseaResNet(nn.Module):
    """AseaResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            asea_ls_init_values: float = None,
            args=None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
            init_values for layerscale: from timm dinov2/so150m2: 1e-5
            from cait def: 1e-4, 24/xxs36 -> 1e-5, s36/m36/m48 -> 1e-6
        """
        super(AseaResNet, self).__init__()

        # reduction related
        self.debugging = getattr(args, 'debugging', True)
        self.new_modules_opt = getattr(args, 'new_modules_opt', True)

        axis_squeeze = getattr(args, 'axis_squeeze', 'mean')
        squeeze_proj = getattr(args, 'squeeze_proj', True)
        axis_conv = getattr(args, 'axis_conv', 3)
        axis_norm = getattr(args, 'axis_norm', True)
        dw_channels = getattr(args, 'red_dw_channels', 1)
        excitation_proj = getattr(args, 'excitation_proj', True)
        proj_v = getattr(args, 'proj_v', True)

        asea_pos = getattr(args, 'asea_pos', 'post_block')
        dropped_axis_fusion = getattr(args, 'dropped_axis_fusion', True)

        keep_rate_var = getattr(args, 'keep_rate_var', [])
        red_train_only = getattr(args, 'red_train_only', False)
        red_random_perturb = getattr(args, 'red_random_perturb', 0.0)
        asea_drop_path = getattr(args, 'asea_drop_path', 1e-5)
        asea_ls_init_values = getattr(args, 'asea_ls_init_values', 1e-5)

        token_ratio = getattr(args, 'keep_rate', [0.5])
        reduction_loc = getattr(args, 'reduction_loc', [2, 4, 6])

        if len(token_ratio) == 1:
            token_ratio = [token_ratio[0] for _ in range(len(reduction_loc))]

        red_axis_start = getattr(args, 'axis_start', '-1')
        token_ratio_full = [[0 for _ in range(nl)] for nl in layers]
        red_axis_full = [[red_axis_start for _ in range(nl)] for nl in layers]
        curr_red_axis = red_axis_start

        for loc, kr in zip(reduction_loc, token_ratio):
            res_curr = loc
            for j, nl in enumerate(layers):
                quotient_int = res_curr // nl
                res = res_curr % nl
                if quotient_int == 0:
                    token_ratio_full[j][res] = kr

                    red_axis_full[j][res] = curr_red_axis
                    if curr_red_axis == '-1':
                        curr_red_axis = '-2'
                    elif curr_red_axis == '-2':
                        curr_red_axis = '-1'

                    break
                elif quotient_int >= 1:
                    res_curr = res_curr - nl

        print('Reduction loc and KR: ', reduction_loc, token_ratio, token_ratio_full, red_axis_full)

        self.reduction_loc = reduction_loc
        self.token_ratio = token_ratio

        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            token_ratio_full,
            red_axis_full,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,

            axis_squeeze=axis_squeeze,
            squeeze_proj=squeeze_proj,
            axis_conv=axis_conv,
            axis_norm=axis_norm,
            dw_channels=dw_channels,
            excitation_proj=excitation_proj,
            proj_v=proj_v,

            asea_pos=asea_pos,
            dropped_axis_fusion=dropped_axis_fusion,

            keep_rate_var=keep_rate_var,
            red_train_only=red_train_only,
            red_random_perturb=red_random_perturb,
            asea_drop_path=asea_drop_path,
            asea_ls_init_values=asea_ls_init_values,

            debugging=self.debugging,

            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        # ifa
        self.ifa_head = getattr(args, 'ifa_head', False)

        if self.ifa_head:
            layers = []
            for name, _ in self.named_modules():
                if ('pre_asea' in name):
                    layers.append(name)

            self.layers = []
            self.features_dic = {}

            self._insert_hooks(layers)

            widths, self.if_channels, layouts = self.get_if_dims(224)
            print('Intermediate layers with channels: ', self.layers, self.if_channels)

            dws_projectors = getattr(args, 'ifa_dws_conv_groups', 2)

            if dws_projectors:
                self.projectors = nn.ModuleList([
                    nn.Sequential(
                        Rearrange('b c -> b c 1'),
                        nn.Conv1d(c, self.num_features, kernel_size=1, stride=1, groups=c),
                        Rearrange('b d 1 -> b d'),
                        nn.LayerNorm(self.num_features),
                    )
                for c in self.if_channels])
            else:
                self.projectors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(c, self.num_features),
                        nn.LayerNorm(self.num_features),
                    )
                for c in self.if_channels])

            del self.fc

            inter_feats = (len(self.reduction_loc) + 1)

            self.ifa_head = nn.Linear(inter_feats * self.num_features, num_classes)

        self.init_weights(zero_init_last=zero_init_last)

    def _log_layer(self,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        self.features_dic[name] = out

    def _insert_hooks(self, layers):
        for name, layer in self.named_modules():
            if layers is not None:
                if name in layers:
                    self.layers += [name]
                    layer.register_forward_hook(partial(self._log_layer, name))
            else:
                self.layers += [name]
                layer.register_forward_hook(partial(self._log_layer, name))

    def get_if_dims(self, image_size):
        self.features_dic = {}

        with torch.no_grad():
            x = torch.rand(2, 3, image_size, image_size)
            x = self.forward_features(x)

            features = list(self.features_dic.values())
            # print([ft.shape for ft in features])

            widths = []
            channels = []
            layouts = []

            for ft in features:
                if len(ft.shape) == 4:
                    b, c, h, w = ft.shape
                    layouts.append('bchw')
                elif len(ft.shape) == 3:
                    b, s, c = ft.shape
                    w = int(s ** 0.5)
                    layouts.append('bsd')
                else:
                    raise NotImplementedError

                widths.append(w)
                channels.append(c)
        return widths, channels, layouts

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {''}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        if self.ifa_head is not False:
            del self.fc

            inter_feats = (len(self.reduction_loc) + 1)

            self.ifa_head = nn.Linear(inter_feats * self.num_features, num_classes)

            return 0

        self.global_pool, self.fc = create_classifier(self.num_features, num_classes, pool_type=global_pool)

    def get_new_module_names(self):
        if self.new_modules_opt is not False:
            return ['asea', 'ls', 'ifa_head', 'fc']
        return ['ifa_head', 'fc']

    def get_reduction_count(self):
        return self.reduction_loc

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier layer.

        Returns:
            Output tensor.
        """
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        if pre_logits:
            return x
        else:
            if self.ifa_head is not False:
                inter_feats = self.features_dic.values()
                inter_feats = [reduce(ft, 'b c h w -> b c', 'mean') for ft in inter_feats]
                inter_feats = [proj(ft) for proj, ft in zip(self.projectors, inter_feats)]
                inter_feats = inter_feats + [x]
                inter_feats = torch.cat(inter_feats, dim=-1)
                x = self.ifa_head(inter_feats)
            else:
                x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """standarize checkpoints"""
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    # print(state_dict.keys())

    if 'module.head.2.weight' in state_dict:
        # supcon
        prefix = 'module.encoder.'
    elif "module.conv1.weight" in state_dict:
        # mocov3 / in1k swav
        prefix = 'module.'
    elif "model.conv1.weight" in state_dict:
        # medical swav
        prefix = "model."
    elif "target_network.encoder.conv1.weight" in state_dict:
        # medical byol
        prefix = "target_network.encoder."

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # print(state_dict.keys())

    for k, v in state_dict.items():
        # if any(kw in k for kw in ['fc', 'head', 'prototypes']):  # Ignore FC layers
        if any(kw in k for kw in ['head', 'prototypes']):  # Ignore FC layers
            continue
        out_dict[k] = v
    # print(out_dict.keys())

    return out_dict


def _create_resnet(variant, pretrained: bool = False, **kwargs) -> AseaResNet:
    _filter_fn = checkpoint_filter_fn

    return build_model_with_cfg(
        AseaResNet,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def _tcfg(url='', **kwargs):
    return _cfg(url=url, **dict({'interpolation': 'bicubic'}, **kwargs))


def _ttcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'test_input_size': (3, 288, 288), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models',
    }, **kwargs))


def _rcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'crop_pct': 0.95, 'test_input_size': (3, 288, 288), 'test_crop_pct': 1.0,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476'
    }, **kwargs))


def _r3cfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'input_size': (3, 160, 160), 'pool_size': (5, 5),
        'crop_pct': 0.95, 'test_input_size': (3, 224, 224), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476',
    }, **kwargs))


def _gcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic',
        'origin_url': 'https://cv.gluon.ai/model_zoo/classification.html',
    }, **kwargs))


default_cfgs = generate_default_cfgs({
    # ResNet and Wide ResNet trained w/ timm (RSB paper and others)
    'asea_resnet10t.c3_in1k': _ttcfg(
        hf_hub_id='timm/resnet10t.c3_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'asea_resnet14t.c3_in1k': _ttcfg(
        hf_hub_id='timm/resnet14t.c3_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'asea_resnet18.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet18.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth'),
    'asea_resnet18.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet18.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth'),
    'asea_resnet18.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet18.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth'),
    'asea_resnet18d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet18d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        first_conv='conv1.0'),
    'asea_resnet34.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet34.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth'),
    'asea_resnet34.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet34.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth'),
    'asea_resnet34.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet34.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth',
        crop_pct=0.95),
    'asea_resnet34.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnet34.bt_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'asea_resnet34d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet34d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        first_conv='conv1.0'),
    'asea_resnet26.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnet26.bt_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth'),
    'asea_resnet26d.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnet26d.bt_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        first_conv='conv1.0'),
    'asea_resnet26t.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet26t.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'asea_resnet50.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet50.a1_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth'),
    'asea_resnet50.a1h_in1k': _rcfg(
        hf_hub_id='timm/resnet50.a1h_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), crop_pct=0.9, test_input_size=(3, 224, 224), test_crop_pct=1.0),
    'asea_resnet50.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet50.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a2_0-a2746f79.pth'),
    'asea_resnet50.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet50.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a3_0-59cae1ef.pth'),
    'asea_resnet50.b1k_in1k': _rcfg(
        hf_hub_id='timm/resnet50.b1k_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b1k-532a802a.pth'),
    'asea_resnet50.b2k_in1k': _rcfg(
        hf_hub_id='timm/resnet50.b2k_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b2k-1ba180c1.pth'),
    'asea_resnet50.c1_in1k': _rcfg(
        hf_hub_id='timm/resnet50.c1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c1-5ba5e060.pth'),
    'asea_resnet50.c2_in1k': _rcfg(
        hf_hub_id='timm/resnet50.c2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c2-d01e05b2.pth'),
    'asea_resnet50.d_in1k': _rcfg(
        hf_hub_id='timm/resnet50.d_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_d-f39db8af.pth'),
    'asea_resnet50.ram_in1k': _ttcfg(
        hf_hub_id='timm/resnet50.ram_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth'),
    'asea_resnet50.am_in1k': _tcfg(
        hf_hub_id='timm/resnet50.am_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_am-6c502b37.pth'),
    'asea_resnet50.ra_in1k': _ttcfg(
        hf_hub_id='timm/resnet50.ra_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ra-85ebb6e5.pth'),
    'asea_resnet50.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnet50.bt_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/rw_resnet50-86acaeed.pth'),
    'asea_resnet50d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet50d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        first_conv='conv1.0'),
    'asea_resnet50d.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet50d.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth',
        first_conv='conv1.0'),
    'asea_resnet50d.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet50d.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a2_0-a3adc64d.pth',
        first_conv='conv1.0'),
    'asea_resnet50d.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet50d.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a3_0-403fdfad.pth',
        first_conv='conv1.0'),
    'asea_resnet50t.untrained': _ttcfg(first_conv='conv1.0'),
    'asea_resnet101.a1h_in1k': _rcfg(
        hf_hub_id='timm/resnet101.a1h_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth'),
    'asea_resnet101.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet101.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth'),
    'asea_resnet101.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet101.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a2_0-6edb36c7.pth'),
    'asea_resnet101.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet101.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a3_0-1db14157.pth'),
    'asea_resnet101d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet101d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'asea_resnet152.a1h_in1k': _rcfg(
        hf_hub_id='timm/resnet152.a1h_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth'),
    'asea_resnet152.a1_in1k': _rcfg(
        hf_hub_id='timm/resnet152.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1_0-2eee8a7a.pth'),
    'asea_resnet152.a2_in1k': _rcfg(
        hf_hub_id='timm/resnet152.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a2_0-b4c6978f.pth'),
    'asea_resnet152.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnet152.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a3_0-134d4688.pth'),
    'asea_resnet152d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet152d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'asea_resnet200.untrained': _ttcfg(),
    'asea_resnet200d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/resnet200d.ra2_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'redwide_resnet50_2.racm_in1k': _ttcfg(
        hf_hub_id='timm/resnet50_2.racm_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth'),

    # torchvision resnet weights
    'asea_resnet18.tv_in1k': _cfg(
        hf_hub_id='timm/resnet18.tv_in1k',
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet34.tv_in1k': _cfg(
        hf_hub_id='timm/resnet34.tv_in1k',
        url='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet35.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet38.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet41.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet44.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet47.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet50.tv_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv_in1k',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet50.tv2_in1k': _cfg(
        hf_hub_id='timm/resnet50.tv2_in1k',
        url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet101.tv_in1k': _cfg(
        hf_hub_id='timm/resnet101.tv_in1k',
        url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet101.tv2_in1k': _cfg(
        hf_hub_id='timm/resnet101.tv2_in1k',
        url='https://download.pytorch.org/models/resnet101-cd907fc2.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet152.tv_in1k': _cfg(
        hf_hub_id='timm/resnet152.tv_in1k',
        url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'asea_resnet152.tv2_in1k': _cfg(
        hf_hub_id='timm/resnet152.tv2_in1k',
        url='https://download.pytorch.org/models/resnet152-f82ba261.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redwide_resnet50_2.tv_in1k': _cfg(
        hf_hub_id='timm/wide_resnet50_2.tv_in1k',
        url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redwide_resnet50_2.tv2_in1k': _cfg(
        hf_hub_id='timm/wide_resnet50_2.tv2_in1k',
        url='https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redwide_resnet101_2.tv_in1k': _cfg(
        hf_hub_id='timm/wide_resnet101_2.tv_in1k',
        url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redwide_resnet101_2.tv2_in1k': _cfg(
        hf_hub_id='timm/wide_resnet101_2.tv2_in1k',
        url='https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    # ResNets w/ alternative norm layers
    'asea_resnet50_gn.a1h_in1k': _ttcfg(
        hf_hub_id='timm/resnet50_gn.a1h_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94),

    # ResNeXt trained in timm (RSB paper and others)
    'redresnext50_32x4d.a1h_in1k': _rcfg(
        hf_hub_id='timm/resnext50_32x4d.a1h_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth'),
    'redresnext50_32x4d.a1_in1k': _rcfg(
        hf_hub_id='timm/resnext50_32x4d.a1_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1_0-b5a91a1d.pth'),
    'redresnext50_32x4d.a2_in1k': _rcfg(
        hf_hub_id='timm/resnext50_32x4d.a2_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a2_0-efc76add.pth'),
    'redresnext50_32x4d.a3_in1k': _r3cfg(
        hf_hub_id='timm/resnext50_32x4d.a3_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a3_0-3e450271.pth'),
    'redresnext50_32x4d.ra_in1k': _ttcfg(
        hf_hub_id='timm/resnext50_32x4d.ra_in1k',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth'),
    'redresnext50d_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnext50d_32x4d.bt_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        first_conv='conv1.0'),
    'redresnext101_32x4d.untrained': _ttcfg(),
    'redresnext101_64x4d.c1_in1k': _rcfg(
        hf_hub_id='timm/resnext101_64x4d.c1_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth'),

    # torchvision ResNeXt weights
    'redresnext50_32x4d.tv_in1k': _cfg(
        hf_hub_id='timm/resnext50_32x4d.tv_in1k',
        url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redresnext101_32x8d.tv_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x8d.tv_in1k',
        url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redresnext101_64x4d.tv_in1k': _cfg(
        hf_hub_id='timm/resnext101_64x4d.tv_in1k',
        url='https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redresnext50_32x4d.tv2_in1k': _cfg(
        hf_hub_id='timm/resnext50_32x4d.tv2_in1k',
        url='https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'redresnext101_32x8d.tv2_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x8d.tv2_in1k',
        url='https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
    #  from https://github.com/facebookresearch/WSL-Images
    #  Please note the CC-BY-NC 4.0 license on these weights, non-commercial use only.
    'redresnext101_32x8d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
        url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'redresnext101_32x16d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x16d.fb_wsl_ig1b_ft_in1k',
        url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'redresnext101_32x32d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x32d.fb_wsl_ig1b_ft_in1k',
        url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'redresnext101_32x48d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x48d.fb_wsl_ig1b_ft_in1k',
        url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),

    #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'asea_resnet18.fb_ssl_yfcc100m_ft_in1k':  _cfg(
        hf_hub_id='timm/resnet18.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'asea_resnet50.fb_ssl_yfcc100m_ft_in1k':  _cfg(
        hf_hub_id='timm/resnet50.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext50_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x8d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x16d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'asea_resnet18.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnet18.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'asea_resnet50.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnet50.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext50_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext50_32x4d.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x4d.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x8d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x8d.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'redresnext101_32x16d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    # ResNets with anti-aliasing / blur pool
    'asea_resnetaa50d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/resnetaa50d.sw_in12k_ft_in1k',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'asea_resnetaa101d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/resnetaa101d.sw_in12k_ft_in1k',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),

    'asea_resnetaa50d.sw_in12k': _ttcfg(
        hf_hub_id='timm/resnetaa50d.sw_in12k',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'asea_resnetaa50d.d_in12k': _ttcfg(
        hf_hub_id='timm/resnetaa50d.d_in12k',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'asea_resnetaa101d.sw_in12k': _ttcfg(
        hf_hub_id='timm/resnetaa101d.sw_in12k',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),

    'asea_resnetblur18.untrained': _ttcfg(),
    'asea_resnetblur50.bt_in1k': _ttcfg(
        hf_hub_id='timm/resnetblur50.bt_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth'),
    'asea_resnetblur50d.untrained': _ttcfg(first_conv='conv1.0'),
    'asea_resnetblur101d.untrained': _ttcfg(first_conv='conv1.0'),
    'asea_resnetaa34d.untrained': _ttcfg(first_conv='conv1.0'),
    'asea_resnetaa50.a1h_in1k': _rcfg(
        hf_hub_id='timm/resnetaa50.a1h_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetaa50_a1h-4cf422b3.pth'),

    # ResNet-RS models
    'asea_resnetrs50.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs50.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth',
        input_size=(3, 160, 160), pool_size=(5, 5), crop_pct=0.91, test_input_size=(3, 224, 224),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs101.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs101.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.94, test_input_size=(3, 288, 288),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs152.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs152.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs200.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs200.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs270.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs270.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 352, 352),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs350.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs350.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0, test_input_size=(3, 384, 384),
        interpolation='bicubic', first_conv='conv1.0'),
    'asea_resnetrs420.tf_in1k': _cfg(
        hf_hub_id='timm/resnetrs420.tf_in1k',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, test_input_size=(3, 416, 416),
        interpolation='bicubic', first_conv='conv1.0'),

    # gluon resnet weights
    'asea_resnet18.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet18.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth'),
    'asea_resnet34.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet34.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth'),
    'asea_resnet50.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet50.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth'),
    'asea_resnet101.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet101.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth'),
    'asea_resnet152.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet152.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth'),
    'asea_resnet50c.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet50c.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth',
        first_conv='conv1.0'),
    'asea_resnet101c.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet101c.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth',
        first_conv='conv1.0'),
    'asea_resnet152c.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet152c.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth',
        first_conv='conv1.0'),
    'asea_resnet50d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet50d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth',
        first_conv='conv1.0'),
    'asea_resnet101d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet101d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth',
        first_conv='conv1.0'),
    'asea_resnet152d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet152d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth',
        first_conv='conv1.0'),
    'asea_resnet50s.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet50s.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth',
        first_conv='conv1.0'),
    'asea_resnet101s.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet101s.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth',
        first_conv='conv1.0'),
    'asea_resnet152s.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnet152s.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth',
        first_conv='conv1.0'),
    'redresnext50_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnext50_32x4d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth'),
    'redresnext101_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnext101_32x4d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth'),
    'redresnext101_64x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/resnext101_64x4d.gluon_in1k',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth'),

    # MIIL
    # https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/models/utils/factory.py
    # https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth
    'asea_resnet50.in21k_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth',
    ),

    # self-supervised resnets
    # SparK
    # https://github.com/keyu-tian/SparK
    # https://drive.google.com/file/d/1H8605HbxGvrsu4x4rIoNr-Wkd7JkxFPQ/view?usp=share_link
    'asea_resnet50.in1k_spark': _cfg(
        url='https://huggingface.co/RexGusto/resnet50_spark/resolve/main/resnet50_1kpretrained_timm_style.pth'
    ),

    # MoCo v3
    # https://github.com/facebookresearch/moco-v3    
    'asea_resnet50.in1k_mocov3': _cfg(
        url='https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar',),
    
    # SupCon
    # https://github.com/HobbitLong/SupContrast
    # https://www.dropbox.com/scl/fi/1ntnvbm5qa25g5o7pj40z/supcon.pth?rlkey=a2orqoh7j1r1qz5hxgsd5hdwv&e=1&dl=0
    'asea_resnet50.in1k_supcon': _cfg(
        url='https://huggingface.co/RexGusto/resnet50_supcon/resolve/main/supcon.pth'
    ),

    # SWAV in1k
    # https://github.com/facebookresearch/swav
    'asea_resnet50.in1k_swav': _cfg(
        url='https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'
    ),

    # SwAV medical
    # https://drive.google.com/uc?export=download&id=11OWRzifq_BXrcFMZ13H0HwS4UGcaiAn_
    'asea_resnet50.medical_swav': _cfg(
        url='https://huggingface.co/RexGusto/resnet50_swav_medical/resolve/main/SwAV_medical.ckpt'
    ),

    # BYOL medical
    # https://drive.google.com/uc?export=download&id=1eBZYl1rXkKJxz42Wu75uzb1kLg8FTv1H
    'asea_resnet50.medical_byol': _cfg(
        url='https://huggingface.co/RexGusto/resnet50_byol_medical/resolve/main/BYOL_medical.ckpt'
    ),

})


@register_model
def asea_resnet10t(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-10-T model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('asea_resnet10t', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet14t(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-14-T model.
    """
    model_args = dict(block=Bottleneck, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('asea_resnet14t', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet18(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('asea_resnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet18d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet18d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet34(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('asea_resnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet34d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet34d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet26(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet('asea_resnet26', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet26t(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('asea_resnet26t', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet26d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet26d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet35(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 3, 1])
    return _create_resnet('asea_resnet35', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet38(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 4, 1])
    return _create_resnet('asea_resnet38', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet41(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 5, 1])
    return _create_resnet('asea_resnet41', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet44(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 1])
    return _create_resnet('asea_resnet44', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet47(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 2])
    return _create_resnet('asea_resnet47', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return _create_resnet('asea_resnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50c(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep')
    return _create_resnet('asea_resnet50c', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50s(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep')
    return _create_resnet('asea_resnet50s', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50t(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('asea_resnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet101(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('asea_resnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet101c(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep')
    return _create_resnet('asea_resnet101c', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet101d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet101s(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep')
    return _create_resnet('asea_resnet101s', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet152(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('asea_resnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet152c(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-152-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep')
    return _create_resnet('asea_resnet152c', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet152d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet152d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet152s(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-152-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep')
    return _create_resnet('asea_resnet152s', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet200(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('asea_resnet200', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet200d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def redwide_resnet50_2(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return _create_resnet('redwide_resnet50_2', pretrained, **dict(model_args, **kwargs))


@register_model
def redwide_resnet101_2(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128)
    return _create_resnet('redwide_resnet101_2', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnet50_gn(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], norm_layer='groupnorm')
    return _create_resnet('asea_resnet50_gn', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext50_32x4d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4)
    return _create_resnet('redresnext50_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext50d_32x4d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('redresnext50d_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext101_32x4d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4)
    return _create_resnet('redresnext101_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext101_32x8d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8)
    return _create_resnet('redresnext101_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext101_32x16d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt-101 32x16d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16)
    return _create_resnet('redresnext101_32x16d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext101_32x32d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt-101 32x32d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32)
    return _create_resnet('redresnext101_32x32d', pretrained, **dict(model_args, **kwargs))


@register_model
def redresnext101_64x4d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4)
    return _create_resnet('redresnext101_64x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetblur18(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d)
    return _create_resnet('asea_resnetblur18', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetblur50(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d)
    return _create_resnet('asea_resnetblur50', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetblur50d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnetblur50d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetblur101d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnetblur101d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetaa34d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-34-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3],  aa_layer=nn.AvgPool2d, stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnetaa34d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetaa50(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50 model with avgpool anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d)
    return _create_resnet('asea_resnetaa50', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetaa50d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnetaa50d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetaa101d(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-101-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('asea_resnetaa101d', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs50(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs50', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs101(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs101', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs152(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs152', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs200(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs200', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs270(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs270', pretrained, **dict(model_args, **kwargs))



@register_model
def asea_resnetrs350(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs350', pretrained, **dict(model_args, **kwargs))


@register_model
def asea_resnetrs420(pretrained: bool = False, **kwargs) -> AseaResNet:
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('asea_resnetrs420', pretrained, **dict(model_args, **kwargs))


if __name__ == '__main__':
    import torch
    from timm import create_model
    from torchprofile import profile_macs

    # try forward pass
    x = torch.rand(1, 3, 224, 224)

    model = create_model('asea_resnet18')
    flops = profile_macs(model, x)
    print('GFLOPs: ', flops / 1e9)

    model = create_model('resnet18')
    flops = profile_macs(model, x)
    print('GFLOPs: ', flops / 1e9)
