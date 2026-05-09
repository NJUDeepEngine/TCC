import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


BRANCHES_DIT = ("attn", "mlp")
BRANCHES_PIXART = ("attn", "cross-attn", "mlp")


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def branch_index(branches: Sequence[str], name: str) -> int:
    try:
        return list(branches).index(name)
    except ValueError as exc:
        raise KeyError(f"unknown TCC branch {name!r}; expected one of {tuple(branches)}") from exc


def make_stale_mask(shape: Tuple[int, int], fresh_indices: Optional[torch.Tensor], device) -> torch.Tensor:
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    if fresh_indices is not None and fresh_indices.numel() > 0:
        mask.scatter_(1, fresh_indices.to(device=device, dtype=torch.long), False)
    return mask


class TokenPoolCollector:
    """
    Collects one tokenpooled vector per sample for each branch/layer.

    The collector is intentionally prompt/class agnostic. It stores a sample axis
    made of labels or prompts and later builds one shared tokenpool TCC pack.
    """

    def __init__(
        self,
        *,
        target_step: int,
        branches: Sequence[str],
        num_layers: int,
        hidden_dim: int,
        cond_batch_size: Optional[int] = None,
        stale_only: bool = True,
        require_cache_step: bool = False,
    ):
        self.target_step = int(target_step)
        self.branches = tuple(branches)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.cond_batch_size = None if cond_batch_size is None else int(cond_batch_size)
        self.stale_only = bool(stale_only)
        self.require_cache_step = bool(require_cache_step)
        self.values: Dict[Tuple[int, int], List[torch.Tensor]] = {
            (b, layer): [] for b in range(len(self.branches)) for layer in range(self.num_layers)
        }

    def record(
        self,
        *,
        step: int,
        layer: int,
        branch: str,
        tensor: torch.Tensor,
        fresh_indices: Optional[torch.Tensor] = None,
        is_cache_step: bool,
    ) -> None:
        if int(step) != self.target_step:
            return
        if tensor.ndim != 3:
            raise ValueError(f"expected [B,T,D] tensor, got shape {tuple(tensor.shape)}")
        if self.require_cache_step and not is_cache_step:
            return
        if tensor.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected hidden_dim={self.hidden_dim}, got {tensor.shape[-1]}")

        b_idx = branch_index(self.branches, branch)
        x = tensor.detach().float()
        if self.cond_batch_size is not None:
            x = x[: self.cond_batch_size]
            if fresh_indices is not None and fresh_indices.shape[0] >= self.cond_batch_size:
                fresh_indices = fresh_indices[: self.cond_batch_size]

        if self.stale_only and is_cache_step:
            mask = make_stale_mask(x.shape[:2], fresh_indices, x.device)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        self.values[(b_idx, int(layer))].append(pooled.cpu())

    def tensor(self) -> torch.Tensor:
        out = torch.empty((len(self.branches), self.num_layers, 0, self.hidden_dim), dtype=torch.float32)
        sample_len = None
        rows = []
        for b_idx in range(len(self.branches)):
            branch_rows = []
            for layer in range(self.num_layers):
                chunks = self.values[(b_idx, layer)]
                if not chunks:
                    raise RuntimeError(f"missing stats for branch={self.branches[b_idx]} layer={layer}")
                layer_tensor = torch.cat(chunks, dim=0).to(dtype=torch.float32)
                if sample_len is None:
                    sample_len = layer_tensor.shape[0]
                elif sample_len != layer_tensor.shape[0]:
                    raise RuntimeError("inconsistent tokenpool sample axis length across branch/layer")
                branch_rows.append(layer_tensor)
            rows.append(torch.stack(branch_rows, dim=0))
        return torch.stack(rows, dim=0) if rows else out

    def tensor_allow_missing(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_len = None
        for chunks in self.values.values():
            if chunks:
                layer_tensor = torch.cat(chunks, dim=0)
                sample_len = layer_tensor.shape[0]
                break
        if sample_len is None:
            out = torch.empty((len(self.branches), self.num_layers, 0, self.hidden_dim), dtype=torch.float32)
            valid = torch.zeros((len(self.branches), self.num_layers), dtype=torch.bool)
            return out, valid

        rows = []
        valid_rows = []
        for b_idx in range(len(self.branches)):
            branch_rows = []
            branch_valid = []
            for layer in range(self.num_layers):
                chunks = self.values[(b_idx, layer)]
                if chunks:
                    layer_tensor = torch.cat(chunks, dim=0).to(dtype=torch.float32)
                    if sample_len != layer_tensor.shape[0]:
                        raise RuntimeError("inconsistent tokenpool sample axis length across branch/layer")
                    branch_rows.append(layer_tensor)
                    branch_valid.append(True)
                else:
                    branch_rows.append(torch.zeros((sample_len, self.hidden_dim), dtype=torch.float32))
                    branch_valid.append(False)
            rows.append(torch.stack(branch_rows, dim=0))
            valid_rows.append(torch.tensor(branch_valid, dtype=torch.bool))
        return torch.stack(rows, dim=0), torch.stack(valid_rows, dim=0)

    def has_any(self) -> bool:
        return any(bool(chunks) for chunks in self.values.values())


def _procrustes(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    ac = a - mu_a
    bc = b - mu_b
    cov = bc.T @ ac
    u, svals, vh = torch.linalg.svd(cov, full_matrices=False)
    r = u @ vh
    denom = (bc * bc).sum().clamp_min(eps)
    scale = svals.sum() / denom
    return mu_a, mu_b, r, scale


def build_tokenpool_pack(
    a_stats: torch.Tensor,
    b_stats: torch.Tensor,
    *,
    branches: Sequence[str],
    target_step: int,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict:
    if a_stats.shape != b_stats.shape:
        raise ValueError(f"A/B stats shape mismatch: {tuple(a_stats.shape)} vs {tuple(b_stats.shape)}")
    num_branches, num_layers, sample_len, hidden_dim = a_stats.shape
    r_out = torch.empty((num_branches, num_layers, hidden_dim, hidden_dim), dtype=torch.float32, device="cpu")
    s_out = torch.empty((num_branches, num_layers), dtype=torch.float32, device="cpu")
    mu_a_out = torch.empty((num_branches, num_layers, hidden_dim), dtype=torch.float32, device="cpu")
    mu_b_out = torch.empty((num_branches, num_layers, hidden_dim), dtype=torch.float32, device="cpu")
    mse_before = torch.empty((num_branches, num_layers), dtype=torch.float32, device="cpu")
    mse_after = torch.empty((num_branches, num_layers), dtype=torch.float32, device="cpu")
    if valid_mask is None:
        valid_mask = torch.ones((num_branches, num_layers), dtype=torch.bool)
    valid_mask = valid_mask.to(dtype=torch.bool, device="cpu")

    for b_idx in range(num_branches):
        for layer in range(num_layers):
            if not bool(valid_mask[b_idx, layer].item()):
                r_out[b_idx, layer].copy_(torch.eye(hidden_dim, dtype=torch.float32))
                s_out[b_idx, layer] = 1.0
                mu_a_out[b_idx, layer].zero_()
                mu_b_out[b_idx, layer].zero_()
                mse_before[b_idx, layer] = 0.0
                mse_after[b_idx, layer] = 0.0
                continue
            a_layer = a_stats[b_idx, layer].to(device=device)
            b_layer = b_stats[b_idx, layer].to(device=device)
            mu_a, mu_b, r, scale = _procrustes(
                a_layer,
                b_layer,
            )
            corrected = mu_a.view(1, -1) + scale * ((b_layer - mu_b.view(1, -1)) @ r)
            mu_a_out[b_idx, layer].copy_(mu_a.cpu())
            mu_b_out[b_idx, layer].copy_(mu_b.cpu())
            r_out[b_idx, layer].copy_(r.cpu())
            s_out[b_idx, layer] = float(scale.item())
            mse_before[b_idx, layer] = float(torch.mean((b_layer - a_layer) ** 2).item())
            mse_after[b_idx, layer] = float(torch.mean((corrected - a_layer) ** 2).item())

    return {
        "step": int(target_step),
        "pool_mode": "tokenpool",
        "branches": tuple(branches),
        "layers": list(range(num_layers)),
        "sample_axis_name": "representative",
        "sample_axis_len": int(sample_len),
        "R": r_out,
        "s": s_out,
        "muA": mu_a_out,
        "muB": mu_b_out,
        "valid_mask": valid_mask,
        "mse_before": mse_before,
        "mse_after": mse_after,
        "mse_improvement_ratio": mse_after / mse_before.clamp_min(1e-12),
    }


@dataclass
class TccApplyState:
    delta: torch.Tensor


class TokenPoolTccCorrector:
    def __init__(
        self,
        *,
        tcc_dir: str,
        branches: Sequence[str],
        alpha: float,
        target_steps: Iterable[int],
        device: torch.device,
        cache_only: bool = True,
        stale_only: bool = True,
    ):
        self.tcc_dir = tcc_dir
        self.branches = tuple(branches)
        self.alpha = float(alpha)
        self.target_steps = set(int(x) for x in target_steps)
        self.device = device
        self.cache_only = bool(cache_only)
        self.stale_only = bool(stale_only)
        self.current_step = None
        self.current_pack = None
        self.apply_calls = 0
        self.corrected_elements = 0
        self.calls_by_step: Dict[int, int] = {}

    def set_step(self, step: int) -> None:
        step = int(step)
        if self.current_step == step:
            return
        self.current_step = step
        self.current_pack = None
        path = os.path.join(self.tcc_dir, f"step_{step:02d}.pt")
        if not os.path.exists(path):
            return
        pack = torch.load(path, map_location="cpu", weights_only=False)
        self.current_pack = {
            "R": pack["R"].to(self.device, dtype=torch.float32),
            "s": pack["s"].to(self.device, dtype=torch.float32),
            "muA": pack["muA"].to(self.device, dtype=torch.float32),
            "muB": pack["muB"].to(self.device, dtype=torch.float32),
            "branches": tuple(pack.get("branches", self.branches)),
        }

    def apply(
        self,
        tensor: torch.Tensor,
        *,
        step: int,
        layer: int,
        branch: str,
        fresh_indices: Optional[torch.Tensor],
        is_cache_step: bool,
    ) -> torch.Tensor:
        if self.alpha <= 0:
            return tensor
        if self.cache_only and not is_cache_step:
            return tensor
        if int(step) not in self.target_steps:
            return tensor
        self.set_step(int(step))
        if self.current_pack is None:
            return tensor

        b_idx = branch_index(self.current_pack["branches"], branch)
        x = tensor.to(dtype=torch.float32)
        r = self.current_pack["R"][b_idx, int(layer)]
        scale = self.current_pack["s"][b_idx, int(layer)]
        mu_a = self.current_pack["muA"][b_idx, int(layer)].view(1, 1, -1)
        mu_b = self.current_pack["muB"][b_idx, int(layer)].view(1, 1, -1)
        corr = mu_a + scale * ((x - mu_b) @ r)
        out = (1.0 - self.alpha) * x + self.alpha * corr

        if self.stale_only and fresh_indices is not None:
            stale = make_stale_mask(x.shape[:2], fresh_indices, x.device)
            out = torch.where(stale.unsqueeze(-1), out, x)
            corrected = int(stale.sum().item()) * int(x.shape[-1])
        else:
            corrected = int(x.numel())
        self.apply_calls += 1
        self.corrected_elements += corrected
        self.calls_by_step[int(step)] = self.calls_by_step.get(int(step), 0) + 1
        return out.to(dtype=tensor.dtype)
