import h5py
import torch


torch.set_default_dtype(torch.float32)

BRANCHES = ["attn", "mlp"]


def parse_range_or_list(spec: str):
    spec = str(spec).strip()
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def infer_prior_shape(h5_path: str, branch: str = "attn"):
    key = f"{branch}_mean"
    with h5py.File(h5_path, "r") as f:
        dset = f[key]
        if dset.ndim not in (3, 4):
            raise RuntimeError(f"{h5_path}:{key} expected 3D/4D, got {dset.ndim}D")
        return int(dset.shape[1]), int(dset.shape[-1])


@torch.no_grad()
def pooled_CxD_from_h5(
    h5_path: str,
    branch: str,
    layer: int,
    *,
    C: int | None = None,
    D: int | None = None,
    chunk_c: int = 20,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Support two layouts:
      1) pooled prior: [L, C, D]
      2) token-wise prior: [L, C, T, D], pooled by mean over token dim
    Return: [C, D]
    """
    key = f"{branch}_mean"

    with h5py.File(h5_path, "r") as f:
        dset = f[key]
        if dset.ndim not in (3, 4):
            raise RuntimeError(f"{h5_path}:{key} expected 3D/4D, got {dset.ndim}D")
        inferred_C = int(dset.shape[1])
        inferred_D = int(dset.shape[-1])
        if C is None:
            C = inferred_C
        if D is None:
            D = inferred_D
        if inferred_C != C or inferred_D != D:
            raise RuntimeError(
                f"{h5_path}:{key} shape mismatch, got {dset.shape}, expected [L,{C},D] or [L,{C},T,{D}]"
            )
        out_cpu = torch.empty((C, D), dtype=torch.float32, device="cpu")

        for c0 in range(0, C, chunk_c):
            c1 = min(C, c0 + chunk_c)
            if dset.ndim == 3:
                x_np = dset[layer, c0:c1, :]
                x = torch.from_numpy(x_np).to(dtype=torch.float32)
            else:
                x_np = dset[layer, c0:c1, :, :]
                x = torch.from_numpy(x_np).to(dtype=torch.float32).mean(dim=1)
            out_cpu[c0:c1].copy_(x)

    return out_cpu.to(device=device, dtype=torch.float32)


@torch.no_grad()
def procrustes_R_and_s(A: torch.Tensor, B: torch.Tensor):
    muA = A.mean(dim=0)
    muB = B.mean(dim=0)
    Ac = A - muA
    Bc = B - muB
    M = Bc.t() @ Ac
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    Brot = Bc @ R
    s_num = (Ac * Brot).sum()
    s_den = (Brot * Brot).sum() + 1e-12
    s = s_num / s_den
    return muA, muB, R, s


@torch.no_grad()
def apply_rs(x: torch.Tensor, muA: torch.Tensor, muB: torch.Tensor, R: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return muA + s * ((x - muB) @ R)


@torch.no_grad()
def compute_truncated_svd(x: torch.Tensor, rank: int):
    """
    x: [C, D]
    return:
        x_mean: [D]
        U: [D, r]       # right singular vector basis
        c_class: [C, r] # centered coefficients, so x_centered ~= c_class @ U.T
    """
    x64 = x.to(dtype=torch.float64)
    x_mean = x64.mean(dim=0, keepdim=True)
    xc = x64 - x_mean

    U_s, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    r = min(rank, S.shape[0])

    U = Vh[:r, :].t().to(dtype=torch.float32)
    c_class = (U_s[:, :r] * S[:r].unsqueeze(0)).to(dtype=torch.float32)
    return x_mean.squeeze(0).to(dtype=torch.float32), U, c_class


@torch.no_grad()
def compute_s_per_class(
    A: torch.Tensor,
    B: torch.Tensor,
    muA: torch.Tensor,
    muB: torch.Tensor,
    R: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    ac = A - muA
    br = (B - muB) @ R
    num = (ac * br).sum(dim=1)
    den = (br * br).sum(dim=1) + eps
    return num / den


@torch.no_grad()
def compute_u_and_c(
    A: torch.Tensor,
    B: torch.Tensor,
    muA: torch.Tensor,
    muB: torch.Tensor,
    R: torch.Tensor,
    s: torch.Tensor,
    rank: int,
):
    x_rs = apply_rs(B, muA, muB, R, s)
    residual = A - x_rs
    res_mean, U, c_class = compute_truncated_svd(residual, rank=rank)
    c_step = c_class.mean(dim=0)
    return U, c_step, c_class, res_mean
