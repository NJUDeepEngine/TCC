import os

import torch


def tcc_forward(B, R, s, muA, muB):
    corr = muA + s * ((B - muB) @ R)
    delta = corr - B
    return corr, delta


def gather_mu(bank, class_idx, shape, device):
    if bank.ndim == 1:
        return bank, None

    # Token-wise shared priors are stored as [T, D]. In this layout the first
    # axis is token rather than class, so class_idx must not be used.
    if bank.ndim == 2 and len(shape) >= 1 and bank.shape[0] == shape[-1]:
        view_shape = [1] * (len(shape) - 1) + [bank.shape[0], bank.shape[1]]
        return bank.view(*view_shape).expand(*shape, bank.shape[1]), None

    shared = bank.mean(dim=0)
    if class_idx is None:
        if len(shape) == 1:
            return shared.unsqueeze(0).expand(shape[0], shared.shape[0]), None
        view_shape = [1] * len(shape) + [shared.shape[0]]
        return shared.view(*view_shape).expand(*shape, shared.shape[0]), None

    idx = torch.as_tensor(class_idx, device=device, dtype=torch.long)
    if idx.ndim == 0:
        if 0 <= int(idx.item()) < bank.shape[0]:
            return bank[idx], torch.tensor(True, device=device)
        return shared, torch.tensor(False, device=device)

    flat_idx = idx.reshape(-1)
    out = torch.empty((flat_idx.numel(), bank.shape[-1]), device=device, dtype=torch.float32)
    valid = (flat_idx >= 0) & (flat_idx < bank.shape[0])
    if valid.any():
        out[valid] = bank[flat_idx[valid]]
    if (~valid).any():
        out[~valid] = shared.unsqueeze(0)

    if tuple(idx.shape) == tuple(shape):
        return out.view(*shape, bank.shape[-1]), valid.view(*shape)

    out = out.view(*shape[:-1], bank.shape[-1]).unsqueeze(-2).expand(*shape, bank.shape[-1])
    valid = valid.view(*shape[:-1]).unsqueeze(-1).expand(*shape)
    return out, valid


def gather_class(bank, class_idx, shape, device):
    idx = torch.as_tensor(class_idx, device=device, dtype=torch.long)
    if idx.ndim == 0:
        return bank[idx]
    if tuple(idx.shape) == tuple(shape):
        return bank[idx.reshape(-1)].view(*shape, bank.shape[-1])
    out = bank[idx.reshape(-1)].view(*shape[:-1], bank.shape[-1])
    return out.unsqueeze(-2).expand(*shape, bank.shape[-1])


def gather_shift(bank, class_idx, shape, device):
    idx = torch.as_tensor(class_idx, device=device, dtype=torch.long)
    if idx.ndim == 0:
        if 0 <= int(idx.item()) < bank.shape[0]:
            return bank[idx]
        return torch.zeros(bank.shape[-1], device=device, dtype=torch.float32)

    flat_idx = idx.reshape(-1)
    out = torch.zeros((flat_idx.numel(), bank.shape[-1]), device=device, dtype=torch.float32)
    valid = (flat_idx >= 0) & (flat_idx < bank.shape[0])
    if valid.any():
        out[valid] = bank[flat_idx[valid]]
    if tuple(idx.shape) == tuple(shape):
        return out.view(*shape, bank.shape[-1])
    out = out.view(*shape[:-1], bank.shape[-1])
    return out.unsqueeze(-2).expand(*shape, bank.shape[-1])


def tcc_bank_forward(B, pack, branch, layer, class_idx, device, prefix=""):
    R = pack[f"{prefix}R"][branch, layer]
    s = pack[f"{prefix}s"][branch, layer]
    muA, valid_mask = gather_mu(pack[f"{prefix}muA"][branch, layer], class_idx, B.shape[:-1], device)
    muB, _ = gather_mu(pack[f"{prefix}muB"][branch, layer], class_idx, B.shape[:-1], device)
    corr, delta = tcc_forward(B, R, s, muA, muB)
    if valid_mask is not None:
        corr = torch.where(valid_mask.unsqueeze(-1), corr, B)
        delta = torch.where(valid_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
    return corr, delta, valid_mask


def blend_from_corr_(B, corr, alpha):
    corr.sub_(B)
    corr.mul_(alpha)
    corr.add_(B)
    return corr


def parse_tcc_window(window):
    if window is None:
        return None
    if isinstance(window, str):
        values = [int(x) for x in window.split(",") if x.strip()]
    else:
        values = [int(x) for x in window]
    if len(values) == 2:
        step_min, step_max = values
        return [step_min, step_max, 0, 10**9]
    raise ValueError("--tcc-window must be step_min,step_max")


class TccCorrector:
    """
    Inference-time corrector for DiT intermediate features.

    mode:
        tcc        : TCC feature correction
        mean_shift : class-wise mean-shift differencing
        projected  : projected-bias differencing

    lowrank:
        if True, add low-rank correction after the base correction.

    alpha:
        float                    -> one scalar alpha
        -1.0                     -> read per-step alpha from step_XX.pt["alpha"]
        tensor [step, layer, 2]  -> alpha(step, layer, branch)
    """

    def __init__(
        self,
        tcc_dir,
        device,
        cuda_id=None,
        preload=False,
        mode="tcc",
        lowrank=False,
        deprecated_lowrank=False,
        alpha=0.5,
        window=(12, 18, 5, 20),
        eta=1.0,
        cache_only=True,
        apply_mode="all",
        apply_cond_only=None,
        lowrank_ridge=1e-4,
        num_steps=20,
        subtract_prev_delta=True,
    ):
        assert mode in ["tcc", "mean_shift", "projected"]

        self.tcc_dir = tcc_dir
        self.device = device
        self.preload = preload
        self.mode = mode
        self.lowrank = lowrank
        self.deprecated_lowrank = bool(deprecated_lowrank)
        self.eta = float(eta)
        self.cache_only = cache_only
        if apply_cond_only is not None:
            apply_mode = "cond_only" if bool(apply_cond_only) else "all"
        assert apply_mode in ["all", "cond_only", "cond_on_uncond"]
        self.apply_mode = apply_mode
        self.apply_on_noncache = not cache_only
        self.lowrank_ridge = float(lowrank_ridge)
        self.num_steps = int(num_steps)
        self.window = parse_tcc_window(window)
        self.subtract_prev_delta = bool(subtract_prev_delta)

        self.use_pack_alpha = False
        self.alpha = alpha
        if torch.is_tensor(alpha):
            self.alpha = alpha.to(dtype=torch.float32, device="cpu")
            if self.window is not None:
                s0, s1, l0, l1 = self.window
                for s in range(self.alpha.shape[0]):
                    for l in range(self.alpha.shape[1]):
                        if not (s0 <= s <= s1 and l0 <= l <= l1):
                            self.alpha[s, l, :] = 0.0
        else:
            self.alpha = float(alpha)
            self.use_pack_alpha = self.alpha == -1.0

        self.cache = {}
        self.G_cache = {}
        self.current_step = None
        self.current_pack = None

        if self.preload:
            if cuda_id is None:
                preload_device = torch.device("cpu")
            else:
                preload_device = torch.device(f"cuda:{cuda_id}")
            for step in range(self.num_steps):
                path = os.path.join(self.tcc_dir, f"step_{step:02d}.pt")
                if os.path.exists(path):
                    self.cache[step] = torch.load(path, map_location=preload_device)

    def set_step(self, step, move_to_device=True):
        if step == self.current_step:
            return

        path = os.path.join(self.tcc_dir, f"step_{step:02d}.pt")
        if self.preload and step in self.cache:
            pack = self.cache[step]
        elif os.path.exists(path):
            pack = torch.load(path, map_location="cpu")
        else:
            self.current_step = step
            self.current_pack = None
            return

        if move_to_device:
            self.current_pack = {}
            for key in [
                "R",
                "s",
                "muA",
                "muB",
                "alpha",
                "alpha_map",
                "delta",
                "scale",
                "U",
                "c_class",
                "res_mean",
                "token_R",
                "token_s",
                "token_muA",
                "token_muB",
                "class_R",
                "class_s",
                "class_muA",
                "class_muB",
            ]:
                if key in pack:
                    self.current_pack[key] = pack[key].to(self.device, dtype=torch.float32, non_blocking=True)
            for key in [
                "pool_mode",
                "sample_axis_name",
                "sample_axis_len",
                "mixed_classpool_ratio",
                "mixed_tokenpool_ratio",
                "token_sample_axis_name",
                "token_sample_axis_len",
                "class_sample_axis_name",
                "class_sample_axis_len",
            ]:
                if key in pack:
                    self.current_pack[key] = pack[key]
        else:
            self.current_pack = pack

        self.current_step = step

    def supports_null_class(self, null_class_idx: int) -> bool:
        if self.current_pack is None:
            return False
        if self.apply_mode == "cond_on_uncond":
            return False
        for key in ("delta", "c_class", "muA", "muB"):
            if (
                key in self.current_pack
                and self.current_pack[key].ndim >= 4
                and self.current_pack[key].shape[2] > int(null_class_idx)
            ):
                return True
        return False

    def apply(self, B, *, branch, layer, deta_pre=None, class_idx=None, is_cache_step=True):
        assert self.current_step is not None
        if self.current_pack is None:
            return B, None
        if self.cache_only and not is_cache_step:
            return B, None
        if self.apply_mode == "cond_only" and B.ndim >= 1 and B.shape[0] % 2 == 0:
            half = B.shape[0] // 2
            prev = None if not isinstance(deta_pre, dict) else {k: (v[:half] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B.shape[0] else v) for k, v in deta_pre.items()}
            prev_mode = self.apply_mode
            self.apply_mode = "all"
            try:
                corr, state = self.apply(B[:half], branch=branch, layer=layer, deta_pre=prev, class_idx=None if class_idx is None else class_idx[:half], is_cache_step=is_cache_step)
            finally:
                self.apply_mode = prev_mode
            return torch.cat([corr, B[half:]], dim=0), state

        if self.window is not None:
            s0, s1, l0, l1 = self.window
            if not (s0 <= self.current_step <= s1 and l0 <= layer <= l1):
                return B, None

        if self.use_pack_alpha:
            alpha_key = "alpha_map" if "alpha_map" in self.current_pack else "alpha"
            if alpha_key not in self.current_pack:
                raise RuntimeError(
                    f"TCC alpha=-1 requested, but step_{self.current_step:02d}.pt does not contain an 'alpha_map' or 'alpha' tensor."
                )
            alpha_bank = self.current_pack[alpha_key]
            if alpha_bank.ndim == 2 and alpha_bank.shape[0] > branch:
                alpha = float(alpha_bank[branch, layer].item())
            elif alpha_bank.ndim == 2 and alpha_bank.shape[-1] > branch:
                alpha = float(alpha_bank[layer, branch].item())
            else:
                raise RuntimeError(
                    f"Unsupported alpha tensor shape {tuple(alpha_bank.shape)} in step_{self.current_step:02d}.pt; "
                    "expected [branch, layer] or [layer, branch]."
                )
        elif torch.is_tensor(self.alpha):
            alpha = float(self.alpha[self.current_step, layer, branch].item())
        else:
            alpha = self.alpha

        if alpha <= 0:
            return B, None

        B = B.to(dtype=torch.float32)
        prev_delta = None
        prev_coeff = None
        prev_bias = None
        prev_step = None
        if torch.is_tensor(deta_pre):
            prev_delta = deta_pre
        elif isinstance(deta_pre, dict):
            prev_delta = deta_pre.get("delta")
            prev_coeff = deta_pre.get("coeff")
            prev_bias = deta_pre.get("bias")
            prev_step = deta_pre.get("step")

        if self.mode == "tcc":
            pool_mode = str(self.current_pack.get("pool_mode", "tokenpool"))
            if pool_mode == "mixed":
                class_ratio = float(self.current_pack.get("mixed_classpool_ratio", 0.5))
                class_ratio = min(max(class_ratio, 0.0), 1.0)
                token_ratio = float(self.current_pack.get("mixed_tokenpool_ratio", 1.0 - class_ratio))
                token_ratio = min(max(token_ratio, 0.0), 1.0)

                token_corr, _, _ = tcc_bank_forward(
                    B,
                    self.current_pack,
                    branch,
                    layer,
                    class_idx,
                    self.device,
                    prefix="token_",
                )
                class_corr, _, _ = tcc_bank_forward(
                    B,
                    self.current_pack,
                    branch,
                    layer,
                    class_idx,
                    self.device,
                    prefix="class_",
                )
                corr = token_ratio * token_corr + class_ratio * class_corr
                delta = corr - B
                valid_mask = None
            else:
                corr, delta, valid_mask = tcc_bank_forward(
                    B,
                    self.current_pack,
                    branch,
                    layer,
                    class_idx,
                    self.device,
                )
            if self.subtract_prev_delta and prev_delta is not None:
                corr = corr - prev_delta
            if valid_mask is not None:
                corr = torch.where(valid_mask.unsqueeze(-1), corr, B)
            out = blend_from_corr_(B, corr, alpha)
            state = delta

        elif self.mode == "mean_shift":
            delta = gather_shift(self.current_pack["delta"][branch, layer], class_idx, B.shape[:-1], self.device)
            corr = B + delta
            if self.subtract_prev_delta and prev_delta is not None:
                corr = corr - prev_delta
            out = blend_from_corr_(B, corr, alpha)
            state = delta

        else:
            R = self.current_pack["R"][branch, layer]
            s = self.current_pack["s"][branch, layer]
            muA, valid_mask = gather_mu(self.current_pack["muA"][branch, layer], class_idx, B.shape[:-1], self.device)
            muB, _ = gather_mu(self.current_pack["muB"][branch, layer], class_idx, B.shape[:-1], self.device)
            M = s * R
            I = torch.eye(M.shape[0], device=self.device, dtype=torch.float32)
            bias = muA - muB
            if valid_mask is not None:
                bias = torch.where(valid_mask.unsqueeze(-1), bias, torch.zeros_like(bias))
            beta = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            if prev_bias is not None:
                beta = torch.clamp((bias * prev_bias).sum() / ((prev_bias * prev_bias).sum() + 1e-12), 0.0, 1.0)
            delta = bias + (B - muB) @ (M - I) if prev_bias is None else bias - beta * prev_bias + (B - muB) @ (M - I)
            if valid_mask is not None:
                delta = torch.where(valid_mask.unsqueeze(-1), delta, torch.zeros_like(delta))
            out = B + alpha * delta
            state = {"step": self.current_step, "delta": delta, "bias": bias}

        if not self.lowrank:
            return out, state

        U = self.current_pack["U"][branch, layer]
        res_mean = self.current_pack.get("res_mean", None)
        if res_mean is not None:
            res_mean = res_mean[branch, layer]
        coeff = gather_class(self.current_pack["c_class"][branch, layer], class_idx, out.shape[:-1], self.device)

        G = None
        if prev_coeff is not None and prev_step is not None:
            key = (prev_step, self.current_step, branch, layer)
            if key in self.G_cache:
                G = self.G_cache[key]
            else:
                prev_path = os.path.join(self.tcc_dir, f"step_{prev_step:02d}.pt")
                if os.path.exists(prev_path):
                    prev_pack = torch.load(prev_path, map_location="cpu")
                    C_prev = prev_pack["c_class"][branch, layer].to(self.device, dtype=torch.float32).t()
                    C_now = self.current_pack["c_class"][branch, layer].t()
                    r = C_prev.shape[0]
                    reg = self.lowrank_ridge * torch.eye(r, device=self.device, dtype=torch.float32)
                    G = (C_now @ C_prev.t()) @ torch.linalg.inv(C_prev @ C_prev.t() + reg)
                    self.G_cache[key] = G

        if prev_coeff is None or G is None:
            coeff_innov = coeff
        elif coeff.ndim == 1:
            coeff_innov = coeff - (G @ prev_coeff)
        else:
            flat_prev = prev_coeff.reshape(-1, prev_coeff.shape[-1])
            coeff_innov = coeff - (G @ flat_prev.t()).t().view_as(coeff)

        use_deprecated_lowrank = self.deprecated_lowrank or (res_mean is None)
        if coeff_innov.ndim == 1:
            delta_lr = coeff_innov @ U.t()
            if not use_deprecated_lowrank:
                delta_lr = res_mean + delta_lr
        else:
            flat = coeff_innov.reshape(-1, coeff_innov.shape[-1])
            delta_lr = (flat @ U.t()).view(*coeff_innov.shape[:-1], U.shape[0])
            if not use_deprecated_lowrank:
                view_shape = [1] * (delta_lr.ndim - 1) + [delta_lr.shape[-1]]
                delta_lr = delta_lr + res_mean.view(*view_shape)

        out = out + self.eta * delta_lr

        if isinstance(state, dict):
            state["coeff"] = coeff
        else:
            state = {"step": self.current_step, "delta": state, "coeff": coeff}
        return out, state
