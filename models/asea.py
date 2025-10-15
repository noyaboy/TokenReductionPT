import math

import torch
import torch.nn as nn

from einops.layers.torch import Reduce
from einops import repeat, rearrange


def complement_idx(idx, dim, axis=-1):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    if axis == -2:
        idx = rearrange(idx, 'b 1 h 1 -> b 1 1 h')

    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))

    if axis == -2:
        compl = rearrange(compl, 'b 1 1 h -> b 1 h 1')

    return compl


def select_tokens_fuse_pruned(x, attn, axis=-1, keep_rate=1.0,
                              dropped_axis_fusion=False, debugging=False):
    idx, compl = None, None

    if debugging:
        print('Pre-reduction shape: ', x.shape)

    B, C, H, W = x.shape

    # select based on top-k
    # rest either drop or fuse (like evit)
    if axis == '-1':
        left_axis = math.floor(keep_rate * W)
        og_axis = W
    elif axis == '-2':
        left_axis = math.floor(keep_rate * H)
        og_axis = H

    if left_axis != og_axis:
        assert left_axis >= 1
        # [B, 1, 1, left_tokens] or [B, 1, left_tokens, 1]
        _, idx = torch.topk(attn, left_axis, dim=int(axis), largest=True, sorted=True)

        if axis == '-1':
            # [B, C, left_tokens, W]
            index = repeat(idx, 'B 1 1 Wl -> B C H Wl', C=C, H=H)

            # [B, C, left_tokens, W]
            x_others = torch.gather(x, dim=-1, index=index)

        elif axis == '-2':
            # [B, C, left_tokens, W]
            index = repeat(idx, 'B 1 Hl 1 -> B C Hl W', C=C, W=W)
            # [B, C, left_tokens, W]
            x_others = torch.gather(x, dim=-2, index=index)


        if dropped_axis_fusion:
            if axis == '-1':
                # compl: # [B, 1, 1, Wd=W-Wl]
                compl = complement_idx(idx, og_axis)

                # [B, C, H, Wd]
                non_topk = torch.gather(x, dim=-1, index=repeat(compl, 'B 1 1 Wd -> B C H Wd', C=C, H=H))

                # [B, 1, 1, Wd]
                non_topk_attn = torch.gather(attn, dim=-1, index=compl)

                # fuse dropped rows by sum across Wd dimension so there's only 1 left
                extra_token = torch.sum(non_topk * non_topk_attn, dim=-1, keepdim=True)

                # combine with original
                x = torch.cat([x_others, extra_token], dim=-1)

                # adjust the idx to account for the fusion token
                idx = torch.cat([idx, torch.ones((B, 1, 1, 1), device=idx.device, dtype=idx.dtype)], dim=-1)

            elif axis == '-2':
                # compl: # [B, 1, Hd=H-Hl, 1]
                compl = complement_idx(idx, og_axis, axis=-2)

                # [B, C, Hd, W]
                non_topk = torch.gather(x, dim=-2, index=repeat(compl, 'B 1 Hd 1 -> B C Hd W', C=C, W=W))

                # [B, 1, Hd, 1]
                non_topk_attn = torch.gather(attn, dim=-2, index=compl)

                # fuse dropped rows by sum across Wd dimension so there's only 1 left
                extra_token = torch.sum(non_topk * non_topk_attn, dim=-2, keepdim=True)

                # combine with original
                x = torch.cat([x_others, extra_token], dim=-2)

                # adjust the idx to account for the fusion token
                idx = torch.cat([idx, torch.ones((B, 1, 1, 1), device=idx.device, dtype=idx.dtype)], dim=-2)

        else:
            x = x_others

    if debugging:
        print('Post-reduction shape: ', x.shape)

    return x, idx, compl


class LayerScale2D(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * rearrange(self.gamma, 'c -> 1 c 1 1')


class AxialSqueezeExcitationAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            axis: str = '-1',
            axis_squeeze: str = 'mean',
            squeeze_proj: bool = True,
            axis_conv: int = 3,
            axis_norm: bool = True,
            dw_channels: int = 1,
            excitation_proj: bool = True,
            proj_v: bool = True,

            axis_size: int = None,            
    ) -> None:
        super().__init__()

        self.pre_asea = nn.Identity()

        inter_dim = dim * dw_channels if dw_channels else dim

        if axis_squeeze in ('mean', 'max'):

            if axis == '-1':
                if squeeze_proj:
                    if axis_conv:
                        self.squeeze_proj = nn.Conv2d(
                            dim,
                            inter_dim,
                            kernel_size=(axis_conv, 1),
                            stride=((axis_conv // 2) + 1, 1),
                            padding=((axis_conv // 2), 0),
                            groups=dim
                        )
                    elif dw_channels:
                        self.squeeze_proj = nn.Conv2d(dim, inter_dim, 1, 1, 0, groups=dim)
                    else:
                        self.squeeze_proj = nn.Conv2d(dim, inter_dim, 1, 1, 0)

                    if axis_norm:
                        self.squeeze_norm = nn.Sequential(
                            nn.BatchNorm2d(inter_dim),
                            nn.ReLU()
                        )
                    
                self.squeeze = Reduce('b c h w -> b c 1 w', axis_squeeze)

            elif axis == '-2':
                if squeeze_proj:
                    if axis_conv:
                        self.squeeze_proj = nn.Conv2d(
                            dim,
                            inter_dim,
                            kernel_size=(1, axis_conv),
                            stride=(1, (axis_conv // 2) + 1),
                            padding=(0, (axis_conv // 2)),
                            groups=dim
                        )
                    elif dw_channels:
                        self.squeeze_proj = nn.Conv2d(dim, inter_dim, 1, 1, 0, groups=dim)
                    else:
                        self.squeeze_proj = nn.Conv2d(dim, inter_dim, 1, 1, 0)

                    if axis_norm:
                        self.squeeze_norm = nn.Sequential(
                            nn.BatchNorm2d(inter_dim),
                            nn.ReLU()
                        )

                self.squeeze = Reduce('b c h w -> b c h 1', axis_squeeze)

            else:
                raise NotImplementedError('axis has to be -1 or -2')


        elif axis_squeeze == 'conv' and axis_size:

            if axis == '-1':
                if dw_channels:
                    self.squeeze = nn.Conv2d(dim, inter_dim, (axis_size, 1), 1, 0, groups=dim)
                else:
                    self.squeeze = nn.Conv2d(dim, inter_dim, (axis_size, 1), 1, 0)
            elif axis == '-2':
                if dw_channels:
                    self.squeeze = nn.Conv2d(dim, inter_dim, (1, axis_size), 1, 0, groups=dim)
                else:
                    self.squeeze = nn.Conv2d(dim, inter_dim, (1, axis_size), 1, 0)
            else:
                raise NotImplementedError('axis has to be -1 or -2')

        else:
            raise NotImplementedError('Need spatial size for axial_squeeze conv')

        if excitation_proj:
            self.excitation = nn.Conv2d(inter_dim, 1, 1, 1, 0)
        else:
            self.excitation = Reduce('b c h w -> b 1 h w', axis_squeeze)

        self.soft = nn.Softmax(dim=-1 if axis == '-1' else -2)

        if proj_v and dw_channels:
            self.w_v = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)
        elif proj_v:
            self.w_v = nn.Conv2d(dim, dim, 1, 1, 0)

        print(self)
        print('ASEA with dim, axis, axis_squeeze, squeeze_proj, excitation_proj, proj_v, dw_channels: ',
              dim, axis, axis_squeeze, squeeze_proj, excitation_proj, proj_v, dw_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for retrieving with hook
        x = self.pre_asea(x)

        # similar to attention / long-kernel attention from VAN
        if hasattr(self, 'w_v'):
            v = self.w_v(x)
        else:
            v = x

        # similar to attention projection (optional)
        if hasattr(self, 'squeeze_proj'):
            # B, C, H, W
            x = self.squeeze_proj(x)

        if hasattr(self, 'squeeze_norm'):
            x = self.squeeze_norm(x)

        # reduces one axis: B, C, 1, W or B, C, H, 1
        x = self.squeeze(x)

        # reduces channels to a scalar (leaving importance of each row only):
        # B, 1, 1, W or B, 1, H, 1
        attn = self.excitation(x)

        # normalize to 1 similar to attention
        attn = self.soft(attn)
        # print(attn.shape, attn[:2])

        # option2 requires projections like vit
        # attn would be basically the attn scores (q@k) and need to produce v independently

        # this step is important because otherwise there is no gradient flow 
        # (top-k is not a differentiable operator but the conv is)
        # reweight the original sequence similar to attention (v could be optional)
        # B, C, H, W
        x = attn * v

        # can follow attention and add multiple heads
        # can also add projection after attn * v

        return x, attn


if __name__ == '__main__':
    asea = AxialSqueezeExcitationAttention(dim=3)
    # asea = AxialSqueezeExcitationAttention(dim=3, axis='-1', axis_squeeze='mean',
    #                                        squeeze_proj=False, excitation_proj=False, proj_v=False)
    x = torch.rand(2, 3, 10, 10)
    h, attn = asea(x)
    x, idx, compl = select_tokens_fuse_pruned(x, attn, '-1', 0.5, True, True)
