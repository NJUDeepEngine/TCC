# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, WindowAttention, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, LabelEmbedder, FinalLayer
from diffusion.utils.logger import get_root_logger
from diffusion.model.cache_functions import global_force_fresh, cache_cutfresh, update_cache, force_init
import json


def _tcc_record(cache_dic, current, branch, tensor, fresh_indices=None, is_cache_step=False):
    collector = cache_dic.get("tcc_collector", None)
    if collector is not None:
        collector.record(
            step=int(current["step"]),
            layer=int(current["layer"]),
            branch=branch,
            tensor=tensor,
            fresh_indices=fresh_indices,
            is_cache_step=bool(is_cache_step),
        )


def _tcc_apply(cache_dic, current, branch, tensor, fresh_indices=None, is_cache_step=False):
    corrector = cache_dic.get("tcc_corrector", None)
    if corrector is None:
        return tensor
    calls_before = getattr(corrector, "apply_calls", 0)
    out = corrector.apply(
        tensor,
        step=int(current["step"]),
        layer=int(current["layer"]),
        branch=branch,
        fresh_indices=fresh_indices,
        is_cache_step=bool(is_cache_step),
    )
    if cache_dic.get("test_FLOPs", False) and getattr(corrector, "apply_calls", 0) > calls_before:
        tokens = int(tensor.shape[0]) * int(tensor.shape[1])
        channels = int(tensor.shape[-1])
        flops = 2.0 * tokens * channels * channels
        cache_dic["tcc_flops"] += flops
        cache_dic["flops"] += flops
    return out


def _linear_flops(batch, tokens, in_features, out_features):
    return 2.0 * batch * tokens * in_features * out_features


def _add_pixart_block_flops(cache_dic, batch, image_tokens, cond_tokens, channels, num_heads, fresh_tokens=None, mlp_tokens=None):
    if not cache_dic.get('test_FLOPs', False):
        return
    n = image_tokens if fresh_tokens is None else fresh_tokens
    m = n if mlp_tokens is None else mlp_tokens
    head_dim = channels // num_heads
    flops = 0.0
    # Match the DiT-ToCa hand-counting style: count the two LayerNorms in each
    # block, but do not charge DiT's per-block adaLN Linear because PixArt uses
    # adaLN-single from the precomputed timestep embedding here.
    flops += 2.0 * batch * image_tokens * channels
    # self-attention is always reused from cache on non-force ToCa steps in this implementation.
    if fresh_tokens is None:
        flops += _linear_flops(batch, image_tokens, channels, 3 * channels)
        flops += batch * num_heads * image_tokens * head_dim
        flops += 4.0 * batch * num_heads * image_tokens * image_tokens * head_dim
        flops += 5.0 * batch * num_heads * image_tokens * image_tokens
        flops += _linear_flops(batch, image_tokens, channels, channels)
    # cross-attention fresh/full path. cond_tokens is already the total
    # number of valid text tokens after PixArt packs the masked batch into a
    # BlockDiagonalMask, so the attention matrix terms should not multiply by
    # batch again.
    flops += _linear_flops(batch, n, channels, channels)
    flops += _linear_flops(1, cond_tokens, channels, 2 * channels)
    flops += batch * num_heads * n * head_dim
    flops += 4.0 * num_heads * n * cond_tokens * head_dim
    flops += 5.0 * num_heads * n * cond_tokens
    flops += _linear_flops(batch, n, channels, channels)
    # MLP fresh/full path.
    hidden = 4 * channels
    flops += _linear_flops(batch, m, channels, hidden)
    flops += 6.0 * batch * m * hidden
    flops += _linear_flops(batch, m, hidden, channels)
    cache_dic['model_flops'] += flops
    cache_dic['flops'] += flops


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.num_heads = num_heads
        self.attn = WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                    input_size=input_size if window_size == 0 else (window_size, window_size),
                                    use_rel_pos=use_rel_pos, **block_kwargs)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, current, cache_dic, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        if not cache_dic.get('use_toca', True):
            _add_pixart_block_flops(cache_dic, B, N, y.shape[1], C, self.num_heads)
            attn_out, _ = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + self.drop_path(gate_msa * attn_out)
            cross_out, _ = self.cross_attn(x, y, mask)
            x = x + cross_out
            mlp_out = self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
            x = x + self.drop_path(gate_mlp * mlp_out)
            return x

        is_force_fresh = global_force_fresh(cache_dic, current)
        if cache_dic.get('tcc_force_full', False):
            is_force_fresh = True
        current['is_force_fresh'] = is_force_fresh
        
        if is_force_fresh: # Compute all tokens, and save them to cache
            _add_pixart_block_flops(cache_dic, B, N, y.shape[1], C, self.num_heads)
            current['module'] = 'attn'
            cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['attn_map'][-1][current['layer']] = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))#.reshape(B, N, C)
            force_init(cache_dic, current, x)
            _tcc_record(cache_dic, current, 'attn', cache_dic['cache'][-1][current['layer']][current['module']], is_cache_step=False)
            x = x + self.drop_path(gate_msa * cache_dic['cache'][-1][current['layer']][current['module']])

            current['module'] = 'cross-attn'
            cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['cross_attn_map'][-1][current['layer']] = self.cross_attn(x, y, mask)
            force_init(cache_dic, current, x)
            _tcc_record(cache_dic, current, 'cross-attn', cache_dic['cache'][-1][current['layer']][current['module']], is_cache_step=False)
            x = x + cache_dic['cache'][-1][current['layer']][current['module']]

            current['module'] = 'mlp'
            cache_dic['cache'][-1][current['layer']][current['module']] = self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
            force_init(cache_dic, current, x)
            _tcc_record(cache_dic, current, 'mlp', cache_dic['cache'][-1][current['layer']][current['module']], is_cache_step=False)
            x = x + self.drop_path(gate_mlp * cache_dic['cache'][-1][current['layer']][current['module']])

        else: 
            current['module'] = 'attn' 
            # no partial computation for attn. if you want to have an exploration, below may help.
            #fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            #fresh_tokens, fresh_attn_map = self.attn(t2i_modulate(self.norm1(fresh_tokens), shift_msa, scale_msa))#.reshape(B, N, C)
            #update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current, fresh_attn_map=fresh_attn_map)
            #cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['attn_map'][-1][current['layer']] = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))#.reshape(B, N, C)
            
            attn_cached = cache_dic['cache'][-1][current['layer']][current['module']]
            attn_cached = _tcc_apply(cache_dic, current, 'attn', attn_cached, fresh_indices=None, is_cache_step=True)
            cache_dic['cache'][-1][current['layer']][current['module']] = attn_cached
            _tcc_record(cache_dic, current, 'attn', attn_cached, fresh_indices=None, is_cache_step=True)
            x = x + self.drop_path(gate_msa * attn_cached)

            if cache_dic.get('cache_mode') == 'fora' or cache_dic.get('fresh_ratio', 0.0) <= 0:
                current['module'] = 'cross-attn'
                cross_cached = cache_dic['cache'][-1][current['layer']][current['module']]
                cross_cached = _tcc_apply(cache_dic, current, 'cross-attn', cross_cached, fresh_indices=None, is_cache_step=True)
                cache_dic['cache'][-1][current['layer']][current['module']] = cross_cached
                _tcc_record(cache_dic, current, 'cross-attn', cross_cached, fresh_indices=None, is_cache_step=True)
                x = x + cross_cached

                current['module'] = 'mlp'
                mlp_cached = cache_dic['cache'][-1][current['layer']][current['module']]
                mlp_cached = _tcc_apply(cache_dic, current, 'mlp', mlp_cached, fresh_indices=None, is_cache_step=True)
                cache_dic['cache'][-1][current['layer']][current['module']] = mlp_cached
                _tcc_record(cache_dic, current, 'mlp', mlp_cached, fresh_indices=None, is_cache_step=True)
                x = x + self.drop_path(gate_mlp * mlp_cached)
                return x

            current['module'] = 'cross-attn'
            fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            cross_fresh_tokens = fresh_tokens.shape[1]
            fresh_tokens, fresh_cross_attn_map = self.cross_attn(fresh_tokens, y, mask)
            update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current, fresh_attn_map=fresh_cross_attn_map)

            cross_cached = cache_dic['cache'][-1][current['layer']][current['module']]
            cross_cached = _tcc_apply(cache_dic, current, 'cross-attn', cross_cached, fresh_indices=fresh_indices, is_cache_step=True)
            cache_dic['cache'][-1][current['layer']][current['module']] = cross_cached
            _tcc_record(cache_dic, current, 'cross-attn', cross_cached, fresh_indices=fresh_indices, is_cache_step=True)
            x = x + cross_cached

            current['module'] = 'mlp'
            fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            _add_pixart_block_flops(cache_dic, B, N, y.shape[1], C, self.num_heads, fresh_tokens=cross_fresh_tokens, mlp_tokens=fresh_tokens.shape[1])
            fresh_tokens = self.mlp(t2i_modulate(self.norm2(fresh_tokens), shift_mlp, scale_mlp))
            update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current)
            
            mlp_cached = cache_dic['cache'][-1][current['layer']][current['module']]
            mlp_cached = _tcc_apply(cache_dic, current, 'mlp', mlp_cached, fresh_indices=fresh_indices, is_cache_step=True)
            cache_dic['cache'][-1][current['layer']][current['module']] = mlp_cached
            _tcc_record(cache_dic, current, 'mlp', mlp_cached, fresh_indices=fresh_indices, is_cache_step=True)
            x = x + self.drop_path(gate_mlp * mlp_cached)

        return x

#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArt(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, pred_sigma=True, drop_path: float = 0., window_size=0, window_block_indexes=None, use_rel_pos=False, caption_channels=4096, lewei_scale=1.0, config=None, model_max_length=120, **kwargs):
        if window_block_indexes is None:
            window_block_indexes = []
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale,

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            PixArtBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                          input_size=(input_size // patch_size, input_size // patch_size),
                          window_size=window_size if i in window_block_indexes else 0,
                          use_rel_pos=use_rel_pos if i in window_block_indexes else False)
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        if config:
            logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
            logger.warning(f"lewei scale: {self.lewei_scale}, base size: {self.base_size}")
        else:
            print(f'Warning: lewei scale: {self.lewei_scale}, base size: {self.base_size}')

    def forward(self, x, timestep, current, cache_dic, y, mask=None, data_info=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for i, block in enumerate(self.blocks):
            current['layer'] = i
            x = auto_grad_checkpoint(block, x, y, t0, current, cache_dic, y_lens)  # (N, T, D) #support grad checkpoint
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, current, cache_dic, y, mask=None, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, current, cache_dic, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, current, cache_dic, y, cfg_scale, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, current, cache_dic, y, mask, kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5), lewei_scale=self.lewei_scale, base_size=self.base_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module()
def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
