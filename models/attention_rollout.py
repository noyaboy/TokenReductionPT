import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat, reduce, rearrange


class TopFeatsSelector(nn.Module):
    def __init__(self, embed_dim, top_k, method='cls', reduction='mean'):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.method = method
        self.reduction = reduction

    def forward(self, feats, attns, i=0):
        if self.method == 'maws':
            attns = [
                attns[f'blocks.{i}.attn.pre_softmax'],
                attns[f'blocks.{i}.attn.attn_drop']
            ]
            attns = calc_maws(attns)

        elif self.method == 'cls':
            attns = attns[f'blocks.{i}.attn.attn_drop']
            attns = reduce(attns, 'b l s1 s2 -> b s1 s2', self.reduction)[:, 0, 1:]

        _, idx = attns.topk(self.top_k, dim=-1, largest=True)
        idx = repeat(idx, 'b k -> b k d', d=self.embed_dim)

        top_feats = torch.gather(feats, 1, idx)
        return top_feats


# @torch.no_grad()
# def calc_cum_attention(attns, l, method='cls', rollout_method='dual'):
def calc_cum_attention(attns, l, method='dar', x=None):
    # print(attns.keys(), l, method)
    # if method == 'cls' and rollout_method == 'dual':
    if method == 'dar':
        dual_attn = [
            attns[f'blocks.{l - 1}.attn.attn_drop'],
            attns[f'blocks.{l}.attn.attn_drop'],
        ]
        # print(len(dual_attn), dual_attn[0].shape)

        crit = dual_attention_rollout(dual_attn)[:, 0, 1:]

    # elif method == 'cls' and rollout_method == 'full':
    elif method == 'ar':
        cum_attn = [
            attns[f'blocks.{i}.attn.attn_drop']
            for i in range(l + 1)
        ]

        curr_shape = cum_attn[-1].shape
        cum_attn = [attn for attn in cum_attn if attn.shape == curr_shape]
        # print(len(cum_attn), cum_attn[0].shape)

        crit = attention_rollout(cum_attn)[:, 0, 1:]

    elif method == 'dmaws':
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
        crit = maws1 * maws2

    elif method == 'maws':
        maws = [
            attns[f'blocks.{l}.attn.pre_softmax'],
            attns[f'blocks.{l}.attn.attn_drop'],
        ]
        crit = calc_maws(maws)

    elif method == 'gls':
        g = x[:, :1]
        l = x[:, 1:]
        crit = F.cosine_similarity(g, l, dim=-1)

    elif method == 'attn':
        attn = attns[f'blocks.{l}.attn.attn_drop'] # B, H, N, N
        attn = attn[:, :, 0, 1:] # [B, H, N-1]
        crit = attn.mean(dim=1)  # [B, N-1]

    elif method == 'glsdmaws':
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
        crit = maws1 * maws2

        g = x[:, :1]
        l = x[:, 1:]
        gls = F.cosine_similarity(g, l, dim=-1)

        crit = crit * gls

    elif method == 'glsmaws':
        maws = [
            attns[f'blocks.{l}.attn.pre_softmax'],
            attns[f'blocks.{l}.attn.attn_drop'],
        ]
        crit = calc_maws(maws)

        g = x[:, :1]
        l = x[:, 1:]
        gls = F.cosine_similarity(g, l, dim=-1)

        crit = crit * gls

    return crit


def calc_attention_map(scores_list):
    if isinstance(scores_list, list) and len(scores_list) == 2:
        scores = dual_attention_rollout(scores_list)
    elif isinstance(scores_list, list) and len(scores_list) > 2:
        scores = attention_rollout(scores_list)
    elif isinstance(scores_list, list) and len(scores_list) == 1:
        scores = reduce(scores_list[0], 'b h s1 s2 -> b s1 s2', 'mean')
    else:
        scores = reduce(scores_list, 'b h s1 s2 -> b s1 s2', 'mean')
    scores = scores[:, 0, 1:]
    attention_map = rearrange(scores, 'b (h w) -> b 1 h w', h=int(scores.shape[-1] ** 0.5))
    return attention_map


def calc_maws(attns, reduction='mean'):
    pre_sm_attn, post_sm_attn = attns
    post_sm_attn = reduce(post_sm_attn, 'b l s1 s2 -> b s1 s2', reduction)[:, 0, 1:]
    pre_sm_attn = reduce(pre_sm_attn, 'b l s1 s2 -> b s1 s2', reduction)
    pre_sm_attn = pre_sm_attn.softmax(dim=-2)[:, 1:, 0]
    maws = pre_sm_attn * post_sm_attn
    return maws


def attention_rollout(scores_list):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = torch.stack(scores_list)

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

    rollout = joint_attentions[-1]

    return rollout


def dual_attention_rollout(scores_list):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    assert len(scores_list) == 2, 'Only works for two layers'
    att_mat = torch.stack(scores_list)

    # Average the attention weights across all heads.
    att_mat = reduce(att_mat, 'l b h s1 s2 -> l b s1 s2', 'mean')

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    dual_attention = torch.matmul(aug_att_mat[-1, :, :1, :], aug_att_mat[0, :, :, :])

    return dual_attention
