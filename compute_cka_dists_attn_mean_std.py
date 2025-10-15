# https://github.com/zhijian-liu/torchprofile
import os
import json
import argparse
from contextlib import suppress
from typing import List, Dict
from warnings import warn
from functools import partial
from statistics import mean, stdev
from itertools import islice

import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from einops import rearrange, reduce, repeat
import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# import models
# import utils
from datasets import build_dataset
from train import get_args_parser, adjust_config, set_seed, set_run_name, build_model



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 15})


MODELS_DIC = {
    # ViT FSL Models
    'deit_tiny_patch16_224.fb_in1k': 'DeiT-T',
    
    'vit_base_patch16_224.orig_in21k': 'ViT',
    'deit_base_patch16_224.fb_in1k': 'DeiT',
    'vit_base_patch16_224_miil.in21k': 'ViT IN21k-P',
    'deit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',
    'deit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',

    'vit_base_patch16_224.orig_in21k_fz': 'ViT',
    'deit_base_patch16_224.fb_in1k_fz': 'DeiT',
    'vit_base_patch16_224_miil.in21k_fz': 'ViT IN21k-P',
    'deit3_base_patch16_224.fb_in22k_ft_in1k_fz': 'DeiT 3 (IN21k)',
    'deit3_base_patch16_224.fb_in1k_fz': 'DeiT 3 (IN1k)',

    # ViT SSL Models
    'vit_base_patch16_224.in1k_mocov3': 'ViT MoCo v3',
    'vit_base_patch16_224.dino': 'ViT DINO',
    'vit_base_patch16_224.mae': 'ViT MAE',
    'vit_base_patch16_clip_224.laion2b': 'ViT CLIP',
    'vit_base_patch16_siglip_224.v2_webli': 'ViT SigLIP v2',

    'vit_base_patch16_224.in1k_mocov3_fz': 'ViT MoCo v3',
    'vit_base_patch16_224.dino_fz': 'ViT DINO',
    'vit_base_patch16_224.mae_fz': 'ViT MAE',
    'vit_base_patch16_clip_224.laion2b_fz': 'ViT CLIP',
    'vit_base_patch16_siglip_224.v2_webli_fz': 'ViT SigLIP v2',
}


DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dafb': 'DAFB',
    'dogs': 'Dogs',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeR1',
    'soyageingr3': 'SoyAgeR3',
    'soyageingr4': 'SoyAgeR4',
    'soyageingr5': 'SoyAgeR5',
    'soyageingr6': 'SoyAgeR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
}


SETTINGS_DIC = {
    'scratch': '(Randomly Initialized)',
    # 'fz': '(Frozen without PETL Modules)',
    'fz': '(Pretrained)',
    'ft': '(Fine-Tuned)',
    'adapter': '(with Adapters)',
}



def add_colorbar(im, aspect=10, pad_fraction=0.2, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


class FeatureMetrics:
    def __init__(self,
                 model: nn.Module,
                 model_name: str = None,
                 model_layers: List[str] = None,
                 device: str ='cpu',
                 setting: str = 'fz',
                 out_size: int = 1,
                 compute_attention_average: bool = False,
                 vit_features_type: str = 'cls',
                 debugging: bool = False):
        """

        :param model: (nn.Module) Neural Network 1
        :param model_name: (str) Name of model 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Setting'] = SETTINGS_DIC.get(setting, setting)

        if model_name is None:
            self.model_info['Name_og'] = model.__repr__().split('(')[0]
        else:
            self.model_info['Name_og'] = model_name
        self.model_info['Name'] = MODELS_DIC.get(self.model_info['Name_og'], self.model_info['Name_og'])

        self.model_info['Layers'] = []

        self.model_features = {}

        if len(list(model.modules())) > 150 and model_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model_layers' parameter. Your CPU/GPU will thank you :)")

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        self.pool = nn.AdaptiveAvgPool2d((out_size, out_size)).to(self.device)

        self.compute_attention_average = compute_attention_average

        self.vit_features_type = vit_features_type

        self.debugging = debugging

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

    def _pool_features(self, feat, batch_flatten=False, pool=False):
        if batch_flatten:
            return feat.flatten(1)

        s = 1

        if pool:
            if len(feat.shape) == 2:
                pooled = feat

            elif len(feat.shape) == 3:
                h = int(feat.shape[1] ** 0.5)

                if h ** 2 == feat.shape[1]:
                    pooled = rearrange(feat, 'b (h w) d -> b d h w', h=h)
                    pooled = self.pool(pooled)
                    s = pooled.shape[-2] * pooled.shape[-1]
                    pooled = rearrange(pooled, 'b c h w -> (b h w) c')  
                else:
                    x_cls, x_others = torch.split(feat, [1, int(h**2)], dim=1)
                    x_others = rearrange(x_others, 'b (h w) d -> b d h w', h=h)
                    x_others = self.pool(x_others)
                    x_others = rearrange(x_others, 'b d h w -> b (h w) d')

                    if self.vit_features_type == 'cls':
                        pooled = x_cls
                    elif self.vit_features_type == 'patches':
                        pooled = x_others
                    else:
                        pooled = torch.cat([x_cls, x_others], dim=1)

                    s = pooled.shape[-2]
                    pooled = rearrange(pooled, 'b s d -> (b s) d')

            elif len(feat.shape) == 4:
                b, c, h, w = feat.shape
                if h != w:
                    feat = rearrange(feat, 'b h w c -> b c h w')
                pooled = self.pool(feat)

                s = feat.shape[-2] * feat.shape[1]
                pooled = rearrange(pooled, 'b c h w -> (b h w) c')

        else:
            if len(feat.shape) == 2:
                pooled = feat
            elif len(feat.shape) == 3:
                s = feat.shape[-2]
                pooled = rearrange(feat, 'b s d -> (b s) d')
            elif len(feat.shape) == 4:
                b, c, h, w = feat.shape
                if h == w:
                    s = feat.shape[-2] * feat.shape[-1]
                    pooled = rearrange(feat, 'b c h w -> (b h w) c')
                else:
                    s = feat.shape[-3] * feat.shape[-2]
                    pooled = rearrange(feat, 'b h w c -> (b h w) c')

        # print(f'{pooled.shape}, batch_flatten: {batch_flatten}, pool: {pool}')
        return pooled, s

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def process_attention(self, x):
        # only process attention of cls token to others
        x = x[:, :, 0, 1:]
        mean = reduce(x, 'b h s2 -> 1', 'mean').squeeze()
        std = torch.std(x)
        return mean, std

    def compare(self,
                dataloader1: DataLoader, max_dataset_iters=None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        """

        self.model_info['Dataset_og'] = dataloader1.dataset.dataset_name
        self.model_info['Dataset'] = DATASETS_DIC.get(self.model_info['Dataset_og'], self.model_info['Dataset_og'])

        layers = self.model_layers if self.model_layers is not None else list(self.model.modules())

        if self.compute_attention_average:
            N = len([layer for layer in layers if 'attn' in layer])
            self.attn_mean = torch.zeros(N, device=self.device)
            self.attn_std = torch.zeros(N, device=self.device)
        else:
            N = len(layers)
            self.hsic_matrix = torch.zeros(N, N, 3)
            self.dist_cum = torch.zeros(N, device=self.device)
            self.dist_cum_norm = torch.zeros(N, device=self.device)
            self.l2_norm = torch.zeros(N, device=self.device)

            self.features_all = {}
            self.labels_all = {}
            self.instances_all = {}
            self.dist_intra_all = []
            self.dist_inter_all = []


        if max_dataset_iters:
            print(f'Slicing dataloader from {len(dataloader1)} to : {max_dataset_iters}')
            dataloader1 = islice(dataloader1, max_dataset_iters)
            num_batches = max_dataset_iters
        else:
            num_batches = len(dataloader1)
        curr_image = 0

        for i, (x1, targets) in tqdm(enumerate(dataloader1), desc="| Comparing features |", total=num_batches):
            self.model_features = {}
            b = x1.shape[0]
            x1 = x1.to(self.device)
            targets = targets.to(self.device)
            instances = torch.arange(curr_image, curr_image + b, device=self.device)
            _ = self.model(x1)

            if self.compute_attention_average:
                self.compare_attn_mean_std(num_batches)
            else:
                self.compare_cka_l2_dist(num_batches, targets, instances)

            curr_image += b

        if not self.compute_attention_average:
            self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                            self.hsic_matrix[:, :, 2].sqrt())

        self.compare_distances()

    def compare_distances(self):
        for i, (layer, feats) in enumerate(self.features_all.items()):
            b = feats.shape[0]
            targets = self.labels_all[layer]
            instances = self.instances_all[layer]
            # print(layer, feats.shape)
 

            # normalize and compute cosine similarity
            feats = F.normalize(feats, p=2, dim=-1)
            similarity = feats @ feats.T
            # print(similarity.shape, torch.max(similarity), torch.min(similarity))

            # cosine distance (0: min/aligned, 1: orthogonal, 2: max/opposite)
            distance = 1 - similarity
            # print(torch.max(distance), torch.min(distance))


            # average distance between features
            # Mask out diagonal
            mask = ~torch.eye(b, dtype=bool, device=self.device)
            # print(mask)

            avg_dist_all = distance[mask].mean()
            # print(distance[mask].shape, distance[mask])
            # print(avg_dist_all)\


            # average distance within same class
            # Pairwise class-equality mask
            same_class = targets.unsqueeze(0) == targets.unsqueeze(1)  # shape (n, n)

            # Exclude diagonal (self pairs)
            mask_same_class = same_class & ~torch.eye(b, dtype=bool, device=self.device)

            # Average within-class distance
            avg_dist_intra_class = distance[mask_same_class].mean()
            # print(distance[mask_same_class].shape, distance[mask_same_class])
            # print(avg_dist_intra_class)
            # self.dist_intra_all[layer] = avg_dist_intra_class
            self.dist_intra_all.append(avg_dist_intra_class)


            # average inter-class (different) distance
            # Pairwise mask for different labels
            diff_class = targets.unsqueeze(0) != targets.unsqueeze(1)

            # Exclude diagonal
            mask_diff_class = diff_class & ~torch.eye(b, dtype=bool, device=self.device)

            avg_dist_inter_class = distance[mask_diff_class].mean()
            # print(distance[mask_diff_class].shape, distance[mask_diff_class])
            # print(avg_dist_inter_class)
            # self.dist_inter_all[layer] = avg_dist_inter_class
            self.dist_inter_all.append(avg_dist_inter_class)


            # average intra-image (instances) distance
            # Pairwise class-equality mask
            # print(instances.shape, instances)
            same_img = instances.unsqueeze(0) == instances.unsqueeze(1)  # shape (n, n)

            # Exclude diagonal (self pairs)
            mask_same_img = same_img & ~torch.eye(b, dtype=bool, device=self.device)

            # Average within-class distance
            avg_dist_intra_instance = distance[mask_same_img].mean()
            # print(distance[mask_same_img].shape, distance[mask_same_img])
            # print(avg_dist_intra_instance)


            # average inter-image distance
            # Pairwise mask for different labels
            diff_img = instances.unsqueeze(0) != instances.unsqueeze(1)

            # Exclude diagonal
            mask_diff_img = diff_img & ~torch.eye(b, dtype=bool, device=self.device)

            avg_dist_inter_instance = distance[mask_diff_img].mean()
            # print(distance[mask_diff_img].shape, distance[mask_diff_img])
            # print(avg_dist_inter_instance)


        self.dist_intra_all = torch.Tensor(self.dist_intra_all)
        self.dist_inter_all = torch.Tensor(self.dist_inter_all)

        return 0

    def compare_attn_mean_std(self, num_batches):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            attn_mean, attn_std = self.process_attention(feat1)
            self.attn_mean[i] += attn_mean / num_batches
            self.attn_std[i] += attn_std / num_batches
        return 0

    def compare_cka_l2_dist(self, num_batches, targets, instances):
        for i, (name1, feat1) in enumerate(self.model_features.items()):
            X = self._pool_features(feat1, pool=False, batch_flatten=True)
            X_pooled, s = self._pool_features(feat1, pool=True)

            # frobenius norm and other distances
            self.l2_norm[i] += torch.norm(X_pooled, p='fro', dim=-1).mean() / num_batches

            dist = torch.cdist(X_pooled, X_pooled, p=2.0)

            dist_avg = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum[i] += dist_avg / num_batches

            dist = (dist - dist.min()) / (dist.max() - dist.min())
            dist_avg_norm = (torch.sum(dist) / torch.nonzero(dist).size(0))
            self.dist_cum_norm[i] += dist_avg_norm / num_batches

            # labels
            targets_batched = repeat(targets, 'b -> (b s)', s=s)
            instances_batched = repeat(instances, 'b -> (b s)', s=s)

            # store features
            if name1 in self.features_all.keys():
                self.features_all[name1] = torch.cat([self.features_all[name1], X_pooled], dim=0)
                self.labels_all[name1] = torch.cat([self.labels_all[name1], targets_batched], dim=0)
                self.instances_all[name1] = torch.cat([self.instances_all[name1], instances_batched], dim=0)
            else:
                self.features_all[name1] = X_pooled
                self.labels_all[name1] = targets_batched
                self.instances_all[name1] = instances_batched

            # cka
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

            for j, (name2, feat2) in enumerate(self.model_features.items()):
                Y = self._pool_features(feat2, pool=False, batch_flatten=True)

                L = Y @ Y.t()
                L.fill_diagonal_(0)

                assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        if self.compute_attention_average:
            return {
                "model_name": self.model_info['Name'],
                "model_name_og": self.model_info['Name_og'],
                "model_layers": self.model_info['Layers'],
                "dataset_name": self.model_info['Dataset'],
                "dataset_name_og": self.model_info['Dataset_og'],
                "setting": self.model_info['Setting'],
                'attn_mean': self.attn_mean,
                'attn_std': self.attn_std,
            }

        return {
            "model_name": self.model_info['Name'],
            "model_name_og": self.model_info['Name_og'],
            "model_layers": self.model_info['Layers'],
            "dataset_name": self.model_info['Dataset'],
            "dataset_name_og": self.model_info['Dataset_og'],
            "setting": self.model_info['Setting'],
            'l2_norm': self.l2_norm,
            "CKA": self.hsic_matrix,
            "dist": self.dist_cum,
            "dist_norm": self.dist_cum_norm,
            "dist_intra": self.dist_intra_all,
            "dist_inter": self.dist_inter_all,
        }

    def plot_cka(self,
                 save_path: str = None,
                 title: str = None,
                 show: bool = False):
        fig, ax = plt.subplots(figsize=(6, 5.25))
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')

        ax.set_xlabel(f"Layers", fontsize=16)
        ax.set_ylabel(f"Layers", fontsize=16)

        labels = range(self.hsic_matrix.shape[0])
        ax.set_xticks(labels)
        ax.set_yticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"CKA on {self.model_info['Dataset']} for {self.model_info['Name']}\n {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        add_colorbar(im)
        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()

    def plot_metrics(self,
                     metric: str = 'norms',
                     save_path: str = None,
                     title: str = None,
                     show: bool = False):
        fig, ax = plt.subplots()

        if metric == 'norms':
            labels = range(self.l2_norm.shape[0])
            ax.bar(labels, self.l2_norm.cpu())
            y_label = 'L2-Norm'
        elif metric == 'dist':
            labels = range(self.dist_cum.shape[0])
            ax.bar(labels, self.dist_cum.cpu())
            y_label = 'L2-Distance'
        elif metric == 'dist_norm':
            labels = range(self.dist_cum_norm.shape[0])
            ax.bar(labels, self.dist_cum_norm.cpu())
            y_label = 'Normalized L2-Dist.'
        elif metric == 'dist_intra':
            labels = range(len(self.dist_intra_all))
            ax.bar(labels, self.dist_intra_all.cpu())
            y_label = 'Intra-Class Cos. Dist.'
        elif metric == 'dist_inter':
            labels = range(len(self.dist_inter_all))
            ax.bar(labels, self.dist_inter_all.cpu())
            y_label = 'Inter-Class Cos. Dist.'
        elif metric == 'attn_mean':
            labels = range(self.attn_mean.shape[0])
            ax.bar(labels, self.attn_mean.cpu())
            y_label = 'Attention Mean'
        elif metric == 'attn_std':
            labels = range(self.attn_std.shape[0])
            ax.bar(labels, self.attn_std.cpu())
            y_label = 'Attention Std.'

        ax.set_xlabel("Layer", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.set_xticks(labels)

        if title is not None:
            ax.set_title(f"{title}", fontsize=17)
        else:
            title = f"{y_label} per Layer on {self.model_info['Dataset']}\n for {self.model_info['Name']} {self.model_info['Setting']}"
            ax.set_title(title, fontsize=17)

        plt.tight_layout(pad=0.25, w_pad=0.25, h_pad=0.25)

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if not self.debugging:
            fn = os.path.splitext(os.path.split(save_path)[-1])[0]
            wandb.log({fn: wandb.Image(fig)})

        if show:
            plt.show()


def calc_cka(results, split='train'):
    name = 'CKA'

    ckas = {}

    results = results[name]

    levels = len(results)
    low = [0, levels // 3]
    mid = [levels // 3, 2 * (levels // 3)]
    high = [2 * (levels // 3), levels]

    for level, interval in zip(['low', 'mid', 'high'], [low, mid, high]):
        name_mean = f'cka_{level}_mean_{split}'
        name_std = f'cka_{level}_std_{split}'
        subset = results[interval[0]:interval[-1], interval[0]:interval[-1]]
        mean_val = subset.mean().item()
        std_val = subset.std().item()
        ckas.update({name_mean: mean_val, name_std: std_val})

    for i, cka in enumerate(results):
        # 1st order derivative of cka with respect to layers
        cka = cka.tolist()
        layer_change = [abs(cka[l + 1] - cka[l]) for l in range(0, len(cka) - 1)]
        try:
            layer_change_mean = mean(layer_change)
            layer_change_std = stdev(layer_change)
        except:
            layer_change_mean = np.nan
            layer_change_std = np.nan

        # 2nd order derivative of cka with respect to layers (change of layer change)
        layer_change_2nd = [abs(layer_change[i + 1] - layer_change[i]) for i in range(0, len(layer_change) - 1)]
        try:
            layer_change_2nd_mean = mean(layer_change_2nd)
            layer_change_2nd_std = stdev(layer_change_2nd)
        except:
            layer_change_2nd_mean = np.nan
            layer_change_2nd_std = np.nan

        ckas.update({
            f'{name.lower()}_change_mean_{i}_{split}': layer_change_mean,
            f'{name.lower()}_change_std_{i}_{split}': layer_change_std,
            f'{name.lower()}_change2_mean_{i}_{split}': layer_change_2nd_mean,
            f'{name.lower()}_change2_std_{i}_{split}': layer_change_2nd_std,
            })

    results = results.fill_diagonal_(0)

    for i, cka in enumerate(results):
        layer_mean = (torch.sum(cka) / torch.nonzero(cka).size(0)).item()
        ckas.update({f'{name.lower()}_{i}_{split}': layer_mean})

    overall_mean = (torch.sum(results) / torch.nonzero(results).size(0)).item()
    ckas.update({f'{name.lower()}_avg_{split}': overall_mean})

    return ckas


def calc_metrics(results, name='l2_norm', split='train'):
    metrics = {}
    for i, metric in enumerate(results[name]):
        metrics.update({f'{name}_{i}_{split}': metric.item()})

    metrics.update({f'{name}_avg_{split}': torch.mean(results[name]).item()})
    return metrics


def calc_attn_mean_std(results, split='train'):
    attns = {}
    for i, (attn_mean, attn_std) in enumerate(zip(results['attn_mean'], results['attn_std'])):
        attns.update({f'attn_mean_{i}_{split}': attn_mean.item()})

        attns.update({f'attn_std_{i}_{split}': attn_std.item()})

    attns.update({f'attn_mean_avg_{split}': torch.mean(results['attn_mean']).item()})
    return attns


def save_results_to_json(args, results_train, results_test):
    # needs to convert tensors (l2_norm, dist, dist_norm, CKA) to list
    if args.compute_attention_average:
        results_train['attn_mean'] = results_train['attn_mean'].tolist()
        results_test['attn_mean'] = results_test['attn_mean'].tolist()
        results_train['attn_std'] = results_train['attn_std'].tolist()
        results_test['attn_std'] = results_test['attn_std'].tolist()
    else:
        results_train['l2_norm'] = results_train['l2_norm'].tolist()
        results_train['dist'] = results_train['dist'].tolist()
        results_train['dist_norm'] = results_train['dist_norm'].tolist()
        results_train['dist_intra'] = results_train['dist_intra'].tolist()
        results_train['dist_inter'] = results_train['dist_inter'].tolist()
        results_train['CKA'] = results_train['CKA'].tolist()

        results_test['l2_norm'] = results_test['l2_norm'].tolist()
        results_test['dist'] = results_test['dist'].tolist()
        results_test['dist_norm'] = results_test['dist_norm'].tolist()
        results_test['dist_intra'] = results_test['dist_intra'].tolist()
        results_test['dist_inter'] = results_test['dist_inter'].tolist()
        results_test['CKA'] = results_test['CKA'].tolist()

    data = {'train': results_train, 'test': results_test} 

    fp = os.path.join(args.output_dir, 'feature_metrics.json')
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)

    return 0


@torch.no_grad()
def compute_metrics(train_loader, test_loader, feature_extractor, amp_autocast, args):
    with amp_autocast():
        feature_extractor.compare(train_loader, args.max_dataset_iters)

        results_train = feature_extractor.export()
        if args.compute_attention_average:
            attn_train = calc_attn_mean_std(results_train, split='train')
            feature_extractor.plot_metrics('attn_mean', os.path.join(args.output_dir, 'attn_mean_train.png'))
            feature_extractor.plot_metrics('attn_std', os.path.join(args.output_dir, 'attn_std_train.png'))
        else:
            if args.compute_attention_cka:
                feature_extractor.plot_cka(os.path.join(args.output_dir, 'attn_cka_train.png'))
            else:
                feature_extractor.plot_cka(os.path.join(args.output_dir, 'cka_train.png'))
            feature_extractor.plot_metrics('norms', os.path.join(args.output_dir, 'norms_train.png'))
            feature_extractor.plot_metrics('dist', os.path.join(args.output_dir, 'dist_train.png'))
            feature_extractor.plot_metrics('dist_norm', os.path.join(args.output_dir, 'dist_norm_train.png'))
            feature_extractor.plot_metrics('dist_intra', os.path.join(args.output_dir, 'dist_intra_train.png'))
            feature_extractor.plot_metrics('dist_inter', os.path.join(args.output_dir, 'dist_inter_train.png'))

            cka_train = calc_cka(results_train, split='train')
            dists_train = calc_metrics(results_train, name='dist', split='train')
            dists_norm_train = calc_metrics(results_train, name='dist_norm', split='train')
            dists_intra_train = calc_metrics(results_train, name='dist_intra', split='train')
            dists_inter_train = calc_metrics(results_train, name='dist_inter', split='train')
            norms_train = calc_metrics(results_train, name='l2_norm', split='train')

        feature_extractor.compare(test_loader, args.max_dataset_iters)

        results_test = feature_extractor.export()
        if args.compute_attention_average:
            feature_extractor.plot_metrics('attn_mean', os.path.join(args.output_dir, 'attn_mean_test.png'))
            feature_extractor.plot_metrics('attn_std', os.path.join(args.output_dir, 'attn_std_test.png'))
            attn_test = calc_attn_mean_std(results_test, split='test')
        else:
            if args.compute_attention_cka:
                feature_extractor.plot_cka(os.path.join(args.output_dir, 'attn_cka_test.png'))
            else:
                feature_extractor.plot_cka(os.path.join(args.output_dir, 'cka_test.png'))
            feature_extractor.plot_metrics('norms', os.path.join(args.output_dir, 'norms_test.png'))
            feature_extractor.plot_metrics('dist', os.path.join(args.output_dir, 'dist_test.png'))
            feature_extractor.plot_metrics('dist_norm', os.path.join(args.output_dir, 'dist_norm_test.png'))
            feature_extractor.plot_metrics('dist_intra', os.path.join(args.output_dir, 'dist_intra_test.png'))
            feature_extractor.plot_metrics('dist_inter', os.path.join(args.output_dir, 'dist_inter_test.png'))

            cka_test = calc_cka(results_test, split='test')
            dists_test = calc_metrics(results_test, name='dist', split='test')
            dists_norm_test = calc_metrics(results_test, name='dist_norm', split='test')
            dists_intra_test = calc_metrics(results_test, name='dist_intra', split='test')
            dists_inter_test = calc_metrics(results_test, name='dist_inter', split='test')
            dists_test = calc_metrics(results_test, name='dist', split='test')
            norms_test = calc_metrics(results_test, name='l2_norm', split='test')

    log_dic = {'setting': args.setting}
    if args.compute_attention_average:
        log_dic.update(attn_train)
        log_dic.update(attn_test)
    else:
        log_dic.update(cka_train)
        log_dic.update(dists_train)
        log_dic.update(dists_norm_train)
        log_dic.update(dists_intra_train)
        log_dic.update(dists_inter_train)
        log_dic.update(norms_train)
        log_dic.update(cka_test)
        log_dic.update(dists_test)
        log_dic.update(dists_norm_test)
        log_dic.update(dists_intra_test)
        log_dic.update(dists_inter_test)
        log_dic.update(dists_test)
        log_dic.update(norms_test)

    return log_dic, results_train, results_test



def get_layers(model, model_name, max_layers=None,
               compute_attention_cka=False, compute_attention_average=False):
    layers = []

    for i, (name, _) in enumerate(model.named_modules()):

        # print(name)

        if compute_attention_average:
            if ('deit' in model_name or 'vit' in model_name) and (any(
                [kw in name for kw in ['attn_drop']])):
                layers.append(name)
        elif compute_attention_cka:
            if ('deit' in model_name or 'vit' in model_name) and (any(
                [kw in name for kw in ['attn_drop']])):
                layers.append(name)
        else:
            if ('deit' in model_name or 'vit' in model_name) and (any(
                [kw in name for kw in ['norm2']])) or name == 'norm':
                layers.append(name)

            elif 'resnet' in model_name and 'bn2' in name:
                layers.append(name)

    if max_layers:
        layers = layers[:max_layers]

    return layers


def setup_environment(args):
    set_seed(args.seed)
    cudnn.benchmark = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # adjust input sizes
    args.resize_size = args.resize_size if args.resize_size else int(args.input_size / 0.875)
    args.test_input_size = args.test_input_size if args.test_input_size else args.input_size
    args.test_resize_size = args.test_resize_size if args.test_resize_size else int(args.test_input_size / 0.875)


    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, args.num_classes = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
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
        print('Loaded checkpoint: ', ret)

    layers = get_layers(
        model, args.model, getattr(args, 'max_layers', None),
        getattr(args, 'compute_attention_cka', False),
        getattr(args, 'compute_attention_average', False),
    )

    feature_extractor = FeatureMetrics(
        model, args.model, layers, args.device, args.setting, args.pool_output_size,
        args.compute_attention_average, args.vit_features_type, args.debugging,
    )

    amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress

    return data_loader_train, data_loader_val, feature_extractor, amp_autocast


def setup_args():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    # feature metrics
    parser.add_argument('--max_dataset_iters', type=int, default=None)
    parser.add_argument('--max_layers', type=int, default=None)
    parser.add_argument('--pool_output_size', type=int, default=1)
    parser.add_argument('--vit_features_type', type=str, default='cls',
                        choices=['cls', 'patches', 'all'])
    parser.add_argument('--compute_attention_average', action='store_true',
                        help='otherwise by def computes cka/l2/distances')
    parser.add_argument('--compute_attention_cka', action='store_true',
                        help='otherwise by def computes norm2 output features')
    parser.add_argument('--cka_all_layers', action='store_true',
                        help='if used then uses all layers of vit for cka analysis')
    parser.set_defaults(output_dir='results_inference', try_fused_attn=False)
    args = parser.parse_args()
    adjust_config(args)
    return args


def main():
    args = setup_args()

    train_loader, test_loader, feature_extractor, amp_autocast = setup_environment(args)

    log_dic, results_train, results_test = compute_metrics(
        train_loader, test_loader, feature_extractor, amp_autocast, args)

    if not args.debugging:
        wandb.log(log_dic)
        wandb.finish()
    else:
        print(log_dic)

    save_results_to_json(args, results_train, results_test)

    return 0


if __name__ == "__main__":
    main()
