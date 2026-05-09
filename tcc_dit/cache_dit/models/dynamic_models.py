# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, reuse_att=None, reuse_mlp=None, *, layer_idx: int, tcc_corrector=None, pre_deta=None, class_idx=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # Raw attention output before the gate.
        if reuse_att is None:
            att_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        else:
            att_out = reuse_att
        att_pre_deta = None if pre_deta is None else pre_deta[0]
        mlp_pre_deta = None if pre_deta is None else pre_deta[1]
        attn_deta_pre = None
        mlp_deta_pre = None
        # apply correction on attn raw output (branch=0)
        if tcc_corrector is not None and (reuse_att is not None or tcc_corrector.apply_on_noncache):
            #att_fornt,att_tail=torch.split(att_out, att_out.shape[0] // 2, dim=0)
            #cond_out = tcc_corrector.apply(att_tail, branch=0, layer=layer_idx)
            #att_out = torch.cat([att_fornt, cond_out], dim=0)
            att_out, attn_deta_pre = tcc_corrector.apply(
                att_out,
                branch=0,
                layer=layer_idx,
                deta_pre=att_pre_deta,
                class_idx=class_idx,
                is_cache_step=(reuse_att is not None),
            )

        x = x + gate_msa.unsqueeze(1) * att_out

        # Raw MLP output before the gate.
        if reuse_mlp is None:
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            mlp_out = reuse_mlp

        # apply correction on mlp raw output (branch=1)
        if tcc_corrector is not None and (reuse_mlp is not None or tcc_corrector.apply_on_noncache):
            #mlp_front,mlp_tail=torch.split(mlp_out, mlp_out.shape[0] // 2, dim=0)
            #cond_out = tcc_corrector.apply(mlp_tail, branch=1, layer=layer_idx)
            #mlp_out=torch.cat([mlp_front, cond_out], dim=0)
            mlp_out, mlp_deta_pre = tcc_corrector.apply(
                mlp_out,
                branch=1,
                layer=layer_idx,
                deta_pre=mlp_pre_deta,
                class_idx=class_idx,
                is_cache_step=(reuse_mlp is not None),
            )

        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x, (att_out, mlp_out),(attn_deta_pre,mlp_deta_pre)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.depth=depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.reuse_policy = "l2c"
        self.fora_interval = 1.0
        self.fora_the_first_half = False

        self.reset()
        
    
    def reset(self, start_timestep=20):
        self.start_timestep = start_timestep
        self.cur_timestep = start_timestep-1
        self.reuse_feature = [None] * self.depth
        self.deta_pre=[None]* self.depth
        self._fora_refresh_offsets = None

    def set_reuse_policy(self, policy="l2c", fora_interval=1, the_first_half=False):
        self.reuse_policy = policy
        self.fora_interval = max(1.0, float(fora_interval))
        self.fora_the_first_half = bool(the_first_half)
        self._fora_refresh_offsets = None

    def _build_fora_refresh_offsets(self):
        offsets = set()
        max_offset = max(0, int(self.start_timestep) - 1)
        k = 0
        while True:
            # Use round-half-up so 2.5 -> 3, matching the intended
            # "cache roughly every 2.5 steps" schedule.
            offset = int(math.floor(k * self.fora_interval + 0.5))
            if offset > max_offset:
                break
            offsets.add(offset)
            k += 1
        offsets.add(0)
        self._fora_refresh_offsets = offsets

    def _fora_current_offset(self):
        return int(self.start_timestep - 1 - self.cur_timestep)

    def _fora_cache_enabled(self):
        if not self.fora_the_first_half:
            return True
        return self._fora_current_offset() < int(math.ceil(self.start_timestep / 2.0))

    def _fora_should_refresh(self):
        if not self._fora_cache_enabled():
            return True
        if self._fora_refresh_offsets is None:
            self._build_fora_refresh_offsets()
        offset = self._fora_current_offset()
        return offset in self._fora_refresh_offsets

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    

    def load_ranking(self, path, num_steps, timestep_map, thres):
        self.rank = [None] * num_steps
        from models.router_models import Router, STE

        act_layer, total_layer = 0, 0
        ckpt = torch.load(path, map_location='cpu')['routers']
        routers = torch.nn.ModuleList([
            Router(2*self.depth) for _ in range(num_steps)
        ])
        routers.load_state_dict(ckpt)
        self.timestep_map =  {timestep: i for i, timestep in enumerate(timestep_map)}

        act_att, act_mlp = 0, 0
        for idx, router in enumerate(routers):
            if idx % 2 == 0:
                self.rank[idx] = STE.apply(router(), thres).nonzero().squeeze(0)
                #print(router(), STE.apply(router(), thres).nonzero())
                total_layer += 2 * self.depth
                act_layer += len(self.rank[idx])
                print(f"TImestep {idx}: Not Reuse: {self.rank[idx].squeeze()}")

                if len(self.rank[idx]) > 0:
                    act_att += sum(1 - torch.remainder(self.rank[idx], 2)).item()
                    act_mlp += sum(torch.remainder(self.rank[idx], 2)).item()
                    
        print(f"Total Activate Layer: {act_layer}/{total_layer}")
        print(f"Total Activate Attention: {act_att}/{total_layer//2}")
        print(f"Total Activate MLP: {act_mlp}/{total_layer//2}")

            
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
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, tcc_corrector=None, is_sample=False,**kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if y is None:
            y = kwargs.get("y", None)
        if tcc_corrector is None:
            tcc_corrector = kwargs.get("tcc_corrector", None)
        assert y is not None, "DiT.forward requires class labels y."
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        timestep = t[0].item()
        router_idx = None
        if hasattr(self, "timestep_map"):
            router_idx = self.timestep_map[timestep]
        y_ids = y
        tcc_class_ids = y_ids
        null_label = self.y_embedder.num_classes
        if y_ids.ndim == 1 and y_ids.numel() % 2 == 0:
            half = y_ids.numel() // 2
            if torch.all(y_ids[half:] == null_label):
                # In CFG sampling the second half is the null label (1000), but
                # older class-wise packs only store real ImageNet classes 0..999.
                # Newer packs may additionally store a null-class prior at index 1000.
                supports_null_class = False
                if tcc_corrector is not None and hasattr(tcc_corrector, "supports_null_class"):
                    supports_null_class = tcc_corrector.supports_null_class(null_label)
                if not supports_null_class:
                    tcc_class_ids = torch.cat([y_ids[:half], y_ids[:half]], dim=0)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
       
        if self.reuse_policy == "l2c" and self.cur_timestep % 2 == 1:
            self.reuse_feature = [None] * len(self.reuse_feature)
        L = len(self.blocks)
        _, T, D = x.shape
        collect_uncond_stats = bool(kwargs.get("collect_uncond_stats", False))
        if not is_sample:
            att_sum  = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
            att_sum2 = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
            mlp_sum  = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
            mlp_sum2 = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)   
            cond_mask = (y_ids != null_label)
            step_count = int(cond_mask.sum().item())
            if collect_uncond_stats:
                uncond_mask = (y_ids == null_label)
                uncond_step_count = int(uncond_mask.sum().item())
                uncond_att_sum = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
                uncond_att_sum2 = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
                uncond_mlp_sum = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
                uncond_mlp_sum2 = torch.zeros(L, T, D, dtype=torch.float32, device=x.device)
        for i, block in enumerate(self.blocks):
            att, mlp = None, None

            if self.reuse_policy == "l2c":
                if self.reuse_feature[i] is not None and 2*i not in self.rank[router_idx] :
                    att = self.reuse_feature[i][0]

                if self.reuse_feature[i] is not None and 2*i+1 not in self.rank[router_idx] :
                    mlp = self.reuse_feature[i][1]
            elif self.reuse_policy == "fora":
                should_refresh = self._fora_should_refresh()
                if self.reuse_feature[i] is not None and not should_refresh:
                    att = self.reuse_feature[i][0]
                    mlp = self.reuse_feature[i][1]
            if self.deta_pre[i]==None:
                pre_deta=None
            else:
                pre_deta=self.deta_pre[i]
            x, reuse_feature ,deta= block(
                x, c,
                reuse_att=att, reuse_mlp=mlp,
                layer_idx=i, tcc_corrector=tcc_corrector, pre_deta=pre_deta, class_idx=tcc_class_ids
            )                 # (N, T, D)
            self.reuse_feature[i] = reuse_feature
            if not deta[0]==None:
                self.deta_pre[i]=deta
            att_out, mlp_out = reuse_feature  # (N,T,D)
            if not is_sample:
                # Cond-only stats: collect the y != 1000 half only.
                att_use = att_out[cond_mask]
                mlp_use = mlp_out[cond_mask]

                # Accumulate sum/sumsq along the batch dimension.
                a = att_use.detach().float()
                m = mlp_use.detach().float()

                att_sum[i]  += a.sum(dim=0)
                att_sum2[i] += (a * a).sum(dim=0)
                mlp_sum[i]  += m.sum(dim=0)
                mlp_sum2[i] += (m * m).sum(dim=0)
                if collect_uncond_stats and uncond_step_count > 0:
                    att_uncond = att_out[uncond_mask].detach().float()
                    mlp_uncond = mlp_out[uncond_mask].detach().float()
                    uncond_att_sum[i] += att_uncond.sum(dim=0)
                    uncond_att_sum2[i] += (att_uncond * att_uncond).sum(dim=0)
                    uncond_mlp_sum[i] += mlp_uncond.sum(dim=0)
                    uncond_mlp_sum2[i] += (mlp_uncond * mlp_uncond).sum(dim=0)
        if not is_sample:
            step_pack = {
                "router_idx": int(router_idx) if router_idx is not None else int(self.start_timestep - 1 - self.cur_timestep),    # 0..num_steps-1
                "count": int(step_count),
                "att_sum": att_sum.cpu(), "att_sum2": att_sum2.cpu(),
                "mlp_sum": mlp_sum.cpu(), "mlp_sum2": mlp_sum2.cpu(),
            }    
            if collect_uncond_stats:
                step_pack.update(
                    {
                        "uncond_count": int(uncond_step_count),
                        "uncond_att_sum": uncond_att_sum.cpu(),
                        "uncond_att_sum2": uncond_att_sum2.cpu(),
                        "uncond_mlp_sum": uncond_mlp_sum.cpu(),
                        "uncond_mlp_sum2": uncond_mlp_sum2.cpu(),
                    }
                )
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)

        self.cur_timestep -= 1
        if not is_sample:
            return x,step_pack
        else:
            return x

    def forward_with_cfg(self, x, t, y, cfg_scale, tcc_corrector=None, is_sample=False,**kwargs):
        if tcc_corrector is None:
            tcc_corrector = kwargs.get("tcc_corrector", None)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        out = self.forward(combined, t, y=y, tcc_corrector=tcc_corrector,is_sample=is_sample, **kwargs)
        if isinstance(out, tuple):
            model_out, step_pack = out
        else:
            model_out, step_pack = out, None

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        if step_pack is None:
            return torch.cat([eps, rest], dim=1)
        else:
            return torch.cat([eps, rest], dim=1), step_pack
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
