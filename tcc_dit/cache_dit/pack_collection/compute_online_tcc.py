import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion import create_diffusion
from download import find_model
from pack_collection.common import BRANCHES, parse_range_or_list, procrustes_R_and_s
from tcc_correct import TccCorrector, parse_tcc_window


class SingleStepCollector:
    """Capture the step pack returned at the end of one dynamic DiT forward."""

    def __init__(self):
        self.pack = None

    def add(self, step_pack):
        self.pack = step_pack


class PlainDiTStepCollector:
    """
    Plain DiT does not return a step pack by itself. Forward hooks collect
    raw attn/mlp outputs and convert them into the same sum/sumsq/count layout
    used by the dynamic collector.
    """

    def __init__(self, model, cond_mask: torch.Tensor):
        self.model = model
        self.cond_mask = cond_mask
        self.handles = []
        self.num_layers = len(model.blocks)
        self.hidden_dim = int(model.pos_embed.shape[-1])
        self.token_len = int(model.pos_embed.shape[1])
        self.count = int(cond_mask.sum().item())
        self.att_sum = None
        self.att_sum2 = None
        self.mlp_sum = None
        self.mlp_sum2 = None

    def _accumulate(self, bank_name: str, bank2_name: str, layer_idx: int, output: torch.Tensor):
        out = output.detach().float()
        cond_out = out[self.cond_mask]
        getattr(self, bank_name)[layer_idx] += cond_out.sum(dim=0)
        getattr(self, bank2_name)[layer_idx] += (cond_out * cond_out).sum(dim=0)

    def _make_hook(self, bank_name: str, bank2_name: str, layer_idx: int):
        def hook(_module, _inputs, output):
            self._accumulate(bank_name, bank2_name, layer_idx, output)

        return hook

    def __enter__(self):
        device = self.model.pos_embed.device
        shape = (self.num_layers, self.token_len, self.hidden_dim)
        self.att_sum = torch.zeros(shape, dtype=torch.float32, device=device)
        self.att_sum2 = torch.zeros_like(self.att_sum)
        self.mlp_sum = torch.zeros_like(self.att_sum)
        self.mlp_sum2 = torch.zeros_like(self.att_sum)
        for layer_idx, block in enumerate(self.model.blocks):
            self.handles.append(block.attn.register_forward_hook(self._make_hook("att_sum", "att_sum2", layer_idx)))
            self.handles.append(block.mlp.register_forward_hook(self._make_hook("mlp_sum", "mlp_sum2", layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self, step: int):
        return {
            "router_idx": int(step),
            "count": int(self.count),
            "att_sum": self.att_sum.to("cpu"),
            "att_sum2": self.att_sum2.to("cpu"),
            "mlp_sum": self.mlp_sum.to("cpu"),
            "mlp_sum2": self.mlp_sum2.to("cpu"),
        }


@dataclass(frozen=True)
class DistContext:
    distributed: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device


@dataclass(frozen=True)
class RuntimePaths:
    out_dir: str
    pack_dir: str
    state_root: str
    shard_root: str

    def state_dir(self, line_name: str, step: int) -> str:
        return os.path.join(self.state_root, f"{line_name}_x{step:02d}")

    def shard_path(self, step: int, rank: int) -> str:
        return os.path.join(self.shard_root, f"step_{step:02d}_rank_{rank:02d}.pt")


@dataclass
class LineState:
    current_step: int
    base_dir: str
    run_dir: str


@dataclass(frozen=True)
class TargetStepPlan:
    target_step: int
    rollback_start_step: int
    next_state_step: int
    next_base_dir: str
    next_run_dir: str
    local_stats_path: str


def setup_distributed(args) -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if args.device.startswith("cuda"):
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(args.device)
            if device.index is not None:
                torch.cuda.set_device(device.index)
    else:
        device = torch.device(args.device)

    return DistContext(
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )


def barrier(dist_ctx: DistContext):
    if dist_ctx.distributed:
        dist.barrier()


def is_main_process(dist_ctx: DistContext) -> bool:
    return dist_ctx.rank == 0


def shard_class_indices(num_classes: int, dist_ctx: DistContext):
    return list(range(dist_ctx.rank, num_classes, dist_ctx.world_size))


def fora_refresh_steps(num_steps: int, interval: float):
    interval = max(1.0, float(interval))
    refresh = set()
    k = 0
    while True:
        offset = int(math.floor(k * interval + 0.5))
        if offset >= num_steps:
            break
        refresh.add(num_steps - 1 - offset)
        k += 1
    return refresh


def l2c_refresh_steps(num_steps: int):
    # In the L2C path, odd reverse steps refresh the cache; even steps may
    # reuse features produced by the previous refresh step.
    return {step for step in range(num_steps) if step % 2 == 1}


def normalize_accelerate_method(method: str) -> str:
    return method


def refresh_steps_for_args(args):
    if args.accelerate_method == "fora":
        return fora_refresh_steps(args.num_sampling_steps, args.fora_interval)
    if args.accelerate_method == "l2c":
        return l2c_refresh_steps(args.num_sampling_steps)
    raise ValueError(f"Unsupported --accelerate-method for online TCC: {args.accelerate_method}")


def rollback_start_for_target(target_step: int, refresh_steps):
    # In reverse sampling order, a target step must replay from the nearest
    # refresh step that can rebuild the cache state needed at the target.
    candidates = [int(step) for step in refresh_steps if int(step) >= int(target_step)]
    if not candidates:
        raise ValueError(f"No refresh/non-cache step found before target step {target_step}.")
    return min(candidates)


def reset_model(model, start_step: int):
    try:
        model.reset(start_timestep=start_step + 1)
    except TypeError:
        model.reset()


def is_dynamic_model(model) -> bool:
    return hasattr(model, "reuse_policy")


def build_model(args, diffusion, mode: str, device: torch.device):
    if mode == "base":
        from models.models import DiT_models
    elif mode == "run":
        from models.dynamic_models import DiT_models
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    if mode == "run":
        model.timestep_map = {int(timestep): i for i, timestep in enumerate(diffusion.timestep_map)}
        if args.accelerate_method == "fora":
            model.set_reuse_policy("fora", args.fora_interval, False)
        elif args.accelerate_method == "l2c":
            if args.path is None:
                raise ValueError("--path is required when --accelerate-method l2c")
            model.load_ranking(args.path, args.num_sampling_steps, diffusion.timestep_map, args.thres)
            model.set_reuse_policy("l2c")
        else:
            raise ValueError(f"Unsupported --accelerate-method: {args.accelerate_method}")
    return model


def build_runtime_paths(out_dir: str) -> RuntimePaths:
    out_dir = os.path.abspath(out_dir)
    paths = RuntimePaths(
        out_dir=out_dir,
        pack_dir=os.path.join(out_dir, "tcc_pack"),
        state_root=os.path.join(out_dir, "states"),
        shard_root=os.path.join(out_dir, "shards"),
    )
    os.makedirs(paths.pack_dir, exist_ok=True)
    os.makedirs(paths.state_root, exist_ok=True)
    os.makedirs(paths.shard_root, exist_ok=True)
    return paths


def batch_path(state_dir: str, class_idx: int) -> str:
    return os.path.join(state_dir, f"class_{class_idx:04d}.pt")


def save_state(state_dir: str, class_idx: int, latent_half: torch.Tensor):
    os.makedirs(state_dir, exist_ok=True)
    torch.save(latent_half.detach().to(dtype=torch.float16, device="cpu"), batch_path(state_dir, class_idx))


def load_state(state_dir: str, class_idx: int, device: torch.device) -> torch.Tensor:
    latent = torch.load(batch_path(state_dir, class_idx), map_location="cpu", weights_only=False)
    return latent.to(device=device, dtype=torch.float32)


def ensure_state_files(state_dir: str, class_indices, stage_name: str):
    missing = [class_idx for class_idx in class_indices if not os.path.isfile(batch_path(state_dir, class_idx))]
    if missing:
        preview = ",".join(str(x) for x in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} latent state file(s) in {state_dir} during {stage_name}. "
            f"Examples: {preview}"
        )


def make_initial_latent(args, class_idx: int) -> torch.Tensor:
    latent_size = args.image_size // 8
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.global_seed) * int(args.num_classes) + int(class_idx))
    return torch.randn(
        (args.total_samples_per_class, 4, latent_size, latent_size),
        generator=generator,
        dtype=torch.float32,
    )


def prepare_initial_states_sharded(args, base_state_dir: str, run_state_dir: str, class_indices, rank: int):
    os.makedirs(base_state_dir, exist_ok=True)
    os.makedirs(run_state_dir, exist_ok=True)
    iterator = tqdm(class_indices, desc=f"Init states [rank {rank}]") if class_indices else []
    for class_idx in iterator:
        latent = make_initial_latent(args, class_idx)
        save_state(base_state_dir, class_idx, latent)
        save_state(run_state_dir, class_idx, latent)


def initialize_line_state(args, paths: RuntimePaths, class_indices, dist_ctx: DistContext) -> LineState:
    start_step = args.num_sampling_steps - 1
    base_dir = paths.state_dir("base", start_step)
    run_dir = paths.state_dir("run", start_step)
    if args.init_states or not (os.path.isdir(base_dir) and os.path.isdir(run_dir)):
        prepare_initial_states_sharded(args, base_dir, run_dir, class_indices, dist_ctx.rank)
    barrier(dist_ctx)
    ensure_state_files(base_dir, class_indices, stage_name=f"initial base_x{start_step:02d}")
    ensure_state_files(run_dir, class_indices, stage_name=f"initial run_x{start_step:02d}")
    return LineState(current_step=start_step, base_dir=base_dir, run_dir=run_dir)


def tokenpooled_cond_mean(step_pack: dict) -> torch.Tensor:
    if step_pack is None:
        raise RuntimeError("Missing step_pack while pooling cond stats.")
    count = max(int(step_pack["count"]), 1)
    attn = step_pack["att_sum"].to(dtype=torch.float32).mean(dim=1) / float(count)
    mlp = step_pack["mlp_sum"].to(dtype=torch.float32).mean(dim=1) / float(count)
    return torch.stack([attn, mlp], dim=0)


def classpooled_cond_tokens(step_pack: dict) -> torch.Tensor:
    if step_pack is None:
        raise RuntimeError("Missing step_pack while pooling cond stats.")
    count = max(int(step_pack["count"]), 1)
    attn = step_pack["att_sum"].to(dtype=torch.float32) / float(count)
    mlp = step_pack["mlp_sum"].to(dtype=torch.float32) / float(count)
    return torch.stack([attn, mlp], dim=0)


def use_tokenpool(args) -> bool:
    return args.tcc_prior_pool_mode in ("tokenpool", "mixed")


def use_classpool(args) -> bool:
    return args.tcc_prior_pool_mode in ("classpool", "mixed")


def iter_latent_chunks(x_half: torch.Tensor, chunk_size: int):
    if x_half.shape[0] <= chunk_size:
        yield x_half
        return
    for start in range(0, x_half.shape[0], chunk_size):
        yield x_half[start : start + chunk_size]


def merge_step_packs(step_packs: list[dict]) -> dict | None:
    if not step_packs:
        return None
    merged = {
        "router_idx": int(step_packs[0]["router_idx"]),
        "count": int(sum(int(pack["count"]) for pack in step_packs)),
        "att_sum": None,
        "att_sum2": None,
        "mlp_sum": None,
        "mlp_sum2": None,
    }
    for key in ("att_sum", "att_sum2", "mlp_sum", "mlp_sum2"):
        merged[key] = sum(pack[key].to(dtype=torch.float32) for pack in step_packs)
    return merged


def build_cfg_batch(args, x_half: torch.Tensor, class_idx: int, device: torch.device):
    n = x_half.shape[0]
    x_full = torch.cat([x_half, x_half], dim=0)
    y_cond = torch.full((n,), class_idx, device=device, dtype=torch.long)
    y_null = torch.full((n,), args.num_classes, device=device, dtype=torch.long)
    y = torch.cat([y_cond, y_null], dim=0)
    return x_full, y


def run_dynamic_single_step(
    args,
    diffusion,
    model,
    x_full: torch.Tensor,
    y: torch.Tensor,
    step: int,
    device: torch.device,
    tcc_corrector=None,
    collect_stats: bool = False,
):
    model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
    if tcc_corrector is not None:
        tcc_corrector.set_step(int(step), move_to_device=True)
        model_kwargs["tcc_corrector"] = tcc_corrector

    collector = SingleStepCollector() if collect_stats else None
    t = torch.full((x_full.shape[0],), int(step), device=device, dtype=torch.long)
    out = diffusion.ddim_sample(
        model.forward_with_cfg,
        x_full,
        t,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        eta=0.0,
        collector=collector,
        is_sample=not collect_stats,
    )
    step_pack = collector.pack if collector is not None else None
    return out, step_pack


def run_plain_single_step(
    args,
    diffusion,
    model,
    x_full: torch.Tensor,
    y: torch.Tensor,
    step: int,
    device: torch.device,
    collect_stats: bool = False,
):
    model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
    t = torch.full((x_full.shape[0],), int(step), device=device, dtype=torch.long)
    if collect_stats:
        cond_mask = (y != args.num_classes)
        with PlainDiTStepCollector(model, cond_mask) as hook_collector:
            out = diffusion.ddim_sample(
                model.forward_with_cfg,
                x_full,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=0.0,
                collector=None,
                is_sample=False,
            )
        return out, hook_collector.finalize(step)

    out = diffusion.ddim_sample(
        model.forward_with_cfg,
        x_full,
        t,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        eta=0.0,
        collector=None,
        is_sample=True,
    )
    return out, None


def step_once(
    args,
    diffusion,
    model,
    x_half: torch.Tensor,
    class_idx: int,
    step: int,
    device: torch.device,
    tcc_corrector=None,
    collect_stats: bool = False,
):
    """
    Unified single-step DDIM entry.

    The input is one class-specific half batch. The function expands it into a
    CFG batch internally: the cond half uses the real class label and the uncond
    half uses the null label.
    """

    n = x_half.shape[0]
    x_full, y = build_cfg_batch(args, x_half, class_idx, device)

    if is_dynamic_model(model):
        out, step_pack = run_dynamic_single_step(
            args,
            diffusion,
            model,
            x_full,
            y,
            step,
            device,
            tcc_corrector=tcc_corrector,
            collect_stats=collect_stats,
        )
    else:
        out, step_pack = run_plain_single_step(
            args,
            diffusion,
            model,
            x_full,
            y,
            step,
            device,
            collect_stats=collect_stats,
        )

    next_half = out["sample"][:n].detach()
    if collect_stats and step_pack is None:
        raise RuntimeError(f"Collector did not receive step stats at step {step} for class {class_idx}.")
    return next_half, step_pack


def advance_state(
    args,
    diffusion,
    model,
    start_latent_half: torch.Tensor,
    class_idx: int,
    start_state_step: int,
    end_state_step: int,
    device: torch.device,
    tcc_corrector=None,
):
    """
    Advance one trajectory from x_start to x_end.

    For example, x19 -> x17 executes reverse steps 19 and 18.
    """

    if end_state_step > start_state_step:
        raise ValueError(f"Cannot advance upward from x{start_state_step:02d} to x{end_state_step:02d}.")

    chunk_outputs = []
    for x_chunk in iter_latent_chunks(start_latent_half, args.batch_size):
        reset_model(model, start_state_step)
        current = x_chunk
        for step in range(start_state_step, end_state_step, -1):
            current, _ = step_once(
                args,
                diffusion,
                model,
                current,
                class_idx,
                step,
                device,
                tcc_corrector=tcc_corrector,
                collect_stats=False,
            )
        chunk_outputs.append(current)
    return torch.cat(chunk_outputs, dim=0)


def run_window_with_target_collect(
    args,
    diffusion,
    model,
    start_latent_half: torch.Tensor,
    class_idx: int,
    start_state_step: int,
    target_step: int,
    device: torch.device,
    tcc_corrector=None,
    collect_target_stats: bool = False,
):
    """
    Replay from the nearest refresh step to rebuild the cache state needed at
    the target step. If the target is itself a refresh step, only that one step
    is executed.
    """

    if start_state_step < target_step:
        raise ValueError(
            f"Expected rollback window start >= target step, "
            f"but got start_state_step={start_state_step}, target_step={target_step}."
        )

    chunk_outputs = []
    target_packs = []
    for x_chunk in iter_latent_chunks(start_latent_half, args.batch_size):
        reset_model(model, start_state_step)
        current = x_chunk
        chunk_target_pack = None
        for step in range(start_state_step, target_step - 1, -1):
            current, maybe_pack = step_once(
                args,
                diffusion,
                model,
                current,
                class_idx,
                step,
                device,
                tcc_corrector=tcc_corrector,
                collect_stats=collect_target_stats and step == target_step,
            )
            if maybe_pack is not None:
                chunk_target_pack = maybe_pack
        chunk_outputs.append(current)
        if chunk_target_pack is not None:
            target_packs.append(chunk_target_pack)

    current = torch.cat(chunk_outputs, dim=0)
    target_pack = merge_step_packs(target_packs)

    if collect_target_stats and target_pack is None:
        raise RuntimeError(
            f"Failed to collect target stats for class {class_idx} at step {target_step} "
            f"from rollback window starting at x{start_state_step:02d}."
        )
    return current, target_pack


@torch.no_grad()
def linear_variant_from_centered(a: torch.Tensor, b: torch.Tensor, variant: str):
    variant = normalize_tcc_pack_variant(variant)
    mu_a = a.mean(dim=0)
    mu_b = b.mean(dim=0)
    ac = a - mu_a
    bc = b - mu_b
    hidden_dim = int(a.shape[-1])

    if variant == "full":
        _, _, r_mat, scale = procrustes_R_and_s(a, b)
    elif variant == "shift_only":
        r_mat = torch.eye(hidden_dim, device=a.device, dtype=torch.float32)
        scale = torch.ones((), device=a.device, dtype=torch.float32)
    elif variant == "scale_shift":
        r_mat = torch.eye(hidden_dim, device=a.device, dtype=torch.float32)
        scale = (ac * bc).sum() / ((bc * bc).sum() + 1e-12)
    else:
        raise ValueError(f"Unsupported --tcc-pack-variant: {variant}")

    return mu_a, mu_b, r_mat, scale


def normalize_tcc_pack_variant(variant: str) -> str:
    if variant == "full":
        return "full"
    return variant


def build_shared_tcc_pack(
    cond_a: torch.Tensor,
    cond_b: torch.Tensor,
    device: torch.device,
    *,
    tcc_pack_variant: str = "full",
    sample_axis_name: str = "class",
):
    """
    cond_a / cond_b: [branch, layer, sample_axis, hidden]
    sample_axis can be either class or token.
    """

    num_branches, num_layers, sample_axis_len, hidden_dim = cond_a.shape
    r_out = torch.empty((num_branches, num_layers, hidden_dim, hidden_dim), dtype=torch.float32, device="cpu")
    s_out = torch.empty((num_branches, num_layers), dtype=torch.float32, device="cpu")
    mu_a_out = torch.empty((num_branches, num_layers, hidden_dim), dtype=torch.float32, device="cpu")
    mu_b_out = torch.empty((num_branches, num_layers, hidden_dim), dtype=torch.float32, device="cpu")

    for branch_idx in range(num_branches):
        for layer_idx in range(num_layers):
            a = cond_a[branch_idx, layer_idx].to(device=device, dtype=torch.float32)
            b = cond_b[branch_idx, layer_idx].to(device=device, dtype=torch.float32)
            mu_a, mu_b, r_mat, scale = linear_variant_from_centered(a, b, tcc_pack_variant)
            mu_a_out[branch_idx, layer_idx].copy_(mu_a.to("cpu"))
            mu_b_out[branch_idx, layer_idx].copy_(mu_b.to("cpu"))
            r_out[branch_idx, layer_idx].copy_(r_mat.to("cpu"))
            s_out[branch_idx, layer_idx] = float(scale.item())

    pack = {
        "R": r_out,
        "s": s_out,
        "muA": mu_a_out,
        "muB": mu_b_out,
        "branches": BRANCHES,
        "layers": list(range(num_layers)),
        "num_classes_saved": int(sample_axis_len),
        "sample_axis_name": str(sample_axis_name),
        "sample_axis_len": int(sample_axis_len),
        "tcc_pack_variant": normalize_tcc_pack_variant(str(tcc_pack_variant)),
    }
    return pack


def build_tokenwise_classpool_pack(
    cond_a: torch.Tensor,
    cond_b: torch.Tensor,
    device: torch.device,
    *,
    tcc_pack_variant: str = "full",
):
    """
    cond_a / cond_b: [branch, layer, token, hidden]

    R/s are still computed with shared Procrustes over token samples. During
    application, token-wise prototypes are needed, so muA/muB are stored as
    [T, D].
    """

    num_branches, num_layers, token_len, hidden_dim = cond_a.shape
    r_out = torch.empty((num_branches, num_layers, hidden_dim, hidden_dim), dtype=torch.float32, device="cpu")
    s_out = torch.empty((num_branches, num_layers), dtype=torch.float32, device="cpu")
    mu_a_out = torch.empty((num_branches, num_layers, token_len, hidden_dim), dtype=torch.float32, device="cpu")
    mu_b_out = torch.empty((num_branches, num_layers, token_len, hidden_dim), dtype=torch.float32, device="cpu")

    for branch_idx in range(num_branches):
        for layer_idx in range(num_layers):
            a = cond_a[branch_idx, layer_idx].to(device=device, dtype=torch.float32)
            b = cond_b[branch_idx, layer_idx].to(device=device, dtype=torch.float32)
            _, _, r_mat, scale = linear_variant_from_centered(a, b, tcc_pack_variant)
            r_out[branch_idx, layer_idx].copy_(r_mat.to("cpu"))
            s_out[branch_idx, layer_idx] = float(scale.item())
            mu_a_out[branch_idx, layer_idx].copy_(a.to("cpu"))
            mu_b_out[branch_idx, layer_idx].copy_(b.to("cpu"))

    return {
        "R": r_out,
        "s": s_out,
        "muA": mu_a_out,
        "muB": mu_b_out,
        "branches": BRANCHES,
        "layers": list(range(num_layers)),
        "num_classes_saved": int(token_len),
        "sample_axis_name": "token",
        "sample_axis_len": int(token_len),
        "tcc_pack_variant": normalize_tcc_pack_variant(str(tcc_pack_variant)),
    }


def build_step_pack(
    args,
    cond_a_tokenpool: torch.Tensor | None,
    cond_b_tokenpool: torch.Tensor | None,
    cond_a_classpool: torch.Tensor | None,
    cond_b_classpool: torch.Tensor | None,
    device: torch.device,
):
    mode = str(args.tcc_prior_pool_mode)
    tcc_pack_variant = normalize_tcc_pack_variant(str(args.tcc_pack_variant))
    if mode == "tokenpool":
        pack = build_shared_tcc_pack(
            cond_a_tokenpool,
            cond_b_tokenpool,
            device,
            tcc_pack_variant=tcc_pack_variant,
            sample_axis_name="class",
        )
        pack["pool_mode"] = "tokenpool"
        return pack

    if mode == "classpool":
        pack = build_tokenwise_classpool_pack(
            cond_a_classpool,
            cond_b_classpool,
            device,
            tcc_pack_variant=tcc_pack_variant,
        )
        pack["pool_mode"] = "classpool"
        return pack

    token_pack = build_shared_tcc_pack(
        cond_a_tokenpool,
        cond_b_tokenpool,
        device,
        tcc_pack_variant=tcc_pack_variant,
        sample_axis_name="class",
    )
    class_pack = build_tokenwise_classpool_pack(
        cond_a_classpool,
        cond_b_classpool,
        device,
        tcc_pack_variant=tcc_pack_variant,
    )
    class_ratio = float(args.tcc_prior_mix_classpool_ratio)
    class_ratio = min(max(class_ratio, 0.0), 1.0)
    token_ratio = 1.0 - class_ratio
    pack = {
        "pool_mode": "mixed",
        "mixed_classpool_ratio": class_ratio,
        "mixed_tokenpool_ratio": token_ratio,
        "branches": BRANCHES,
        "layers": token_pack["layers"],
        "token_sample_axis_name": token_pack["sample_axis_name"],
        "token_sample_axis_len": token_pack["sample_axis_len"],
        "class_sample_axis_name": class_pack["sample_axis_name"],
        "class_sample_axis_len": class_pack["sample_axis_len"],
        "token_R": token_pack["R"],
        "token_s": token_pack["s"],
        "token_muA": token_pack["muA"],
        "token_muB": token_pack["muB"],
        "class_R": class_pack["R"],
        "class_s": class_pack["s"],
        "class_muA": class_pack["muA"],
        "class_muB": class_pack["muB"],
        "tcc_pack_variant": normalize_tcc_pack_variant(str(tcc_pack_variant)),
    }
    return pack


def save_meta(args, out_dir: str, target_steps, refresh_steps):
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "online_tcc",
        "tcc_prior_trajectory": args.tcc_prior_trajectory,
        "model": args.model,
        "ckpt": args.ckpt,
        "image_size": args.image_size,
        "num_sampling_steps": args.num_sampling_steps,
        "batch_size_per_forward": args.batch_size,
        "samples_per_class": args.total_samples_per_class,
        "num_classes": args.num_classes,
        "cfg_scale": args.cfg_scale,
        "accelerate_method": args.accelerate_method,
        "fora_interval": args.fora_interval,
        "l2c_router_path": args.path,
        "l2c_router_thres": args.thres,
        "refresh_steps": sorted(int(x) for x in refresh_steps),
        "target_steps_desc": [int(x) for x in target_steps],
        "target_step_alpha": args.tcc_alpha,
        "tcc_window": parse_tcc_window(args.tcc_window),
        "tcc_apply_mode": "all",
        "null_class_stats": False,
        "cache_only": True,
        "subtract_prev_delta": False,
        "tcc_pack_variant": str(args.tcc_pack_variant),
        "tcc_prior_pool_mode": str(args.tcc_prior_pool_mode),
        "tcc_prior_mix_classpool_ratio": float(args.tcc_prior_mix_classpool_ratio)
        if args.tcc_prior_pool_mode == "mixed"
        else None,
        "recommended_sample_args": {"tcc_alpha": float(args.tcc_alpha)},
        "tcc_pack_layout": {
            "tokenpool": "shared_mu_classwise_tokenpool",
            "classpool": "shared_tokenbank_classpool",
            "mixed": "mixed_tokenpool_and_classpool",
        }[str(args.tcc_prior_pool_mode)],
        "collector_mode": "rollback_to_previous_refresh_step",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def make_tcc_corrector(args, pack_dir: str, device: torch.device) -> TccCorrector:
    return TccCorrector(
        tcc_dir=pack_dir,
        device=device,
        cuda_id=device.index,
        preload=False,
        mode="tcc",
        alpha=args.tcc_alpha,
        window=args.tcc_window,
        cache_only=True,
        apply_mode="all",
        num_steps=args.num_sampling_steps,
        subtract_prev_delta=False,
    )


def validate_args(args, target_steps):
    args.batch_size = int(args.batch_size)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    args.total_samples_per_class = int(args.samples) if args.samples is not None else int(args.batch_size)
    if args.total_samples_per_class <= 0:
        raise ValueError("--samples must be a positive integer")
    if args.total_samples_per_class < args.batch_size:
        args.batch_size = args.total_samples_per_class

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available.")

    if not target_steps:
        raise ValueError("No target steps parsed from --target-steps")

    if args.accelerate_method == "l2c" and args.path is None:
        raise ValueError("--path is required when --accelerate-method l2c")

    if float(args.tcc_alpha) < 0.0:
        raise ValueError("--tcc-alpha must be non-negative")
    if args.tcc_prior_trajectory not in ("trajectory_consistent", "one_shot"):
        raise ValueError("--tcc-prior-trajectory must be trajectory_consistent or one_shot")
    if args.tcc_prior_pool_mode not in ("tokenpool", "classpool", "mixed"):
        raise ValueError("--tcc-prior-pool-mode must be one of: tokenpool, classpool, mixed")
    if args.tcc_pack_variant not in ("full", "shift_only", "scale_shift"):
        raise ValueError("--tcc-pack-variant must be one of: full, shift_only, scale_shift")
    if not 0.0 <= float(args.tcc_prior_mix_classpool_ratio) <= 1.0:
        raise ValueError("--tcc-prior-mix-classpool-ratio must lie in [0, 1]")


def make_target_step_plan(
    paths: RuntimePaths,
    target_step: int,
    refresh_steps,
    dist_ctx: DistContext,
) -> TargetStepPlan:
    rollback_start_step = rollback_start_for_target(target_step, refresh_steps)
    next_state_step = target_step - 1
    return TargetStepPlan(
        target_step=target_step,
        rollback_start_step=rollback_start_step,
        next_state_step=next_state_step,
        next_base_dir=paths.state_dir("base", next_state_step),
        next_run_dir=paths.state_dir("run", next_state_step),
        local_stats_path=paths.shard_path(target_step, dist_ctx.rank),
    )


def ensure_valid_target_step(line_state: LineState, plan: TargetStepPlan, refresh_steps):
    if plan.rollback_start_step not in refresh_steps:
        raise ValueError(
            f"Target step {plan.target_step} does not have a valid preceding refresh/non-cache step "
            f"x{plan.rollback_start_step:02d}."
        )


def advance_lines_to_rollback_start(
    args,
    diffusion,
    base_model,
    run_model,
    line_state: LineState,
    plan: TargetStepPlan,
    class_indices,
    dist_ctx: DistContext,
    history_tcc: TccCorrector,
    paths: RuntimePaths,
):
    if line_state.current_step == plan.rollback_start_step:
        return line_state

    target_base_dir = paths.state_dir("base", plan.rollback_start_step)
    target_run_dir = paths.state_dir("run", plan.rollback_start_step)

    if line_state.current_step < plan.rollback_start_step:
        # Handle the case where a cache step immediately follows a refresh step.
        # For example, target 17 can run directly from x17 to x16, but target 16
        # still needs to replay from saved x17 to rebuild the cache.
        ensure_state_files(target_base_dir, class_indices, stage_name=f"rewind base_x{plan.rollback_start_step:02d}")
        ensure_state_files(target_run_dir, class_indices, stage_name=f"rewind run_x{plan.rollback_start_step:02d}")
        return LineState(current_step=plan.rollback_start_step, base_dir=target_base_dir, run_dir=target_run_dir)

    os.makedirs(target_base_dir, exist_ok=True)
    os.makedirs(target_run_dir, exist_ok=True)

    iterator = tqdm(class_indices, desc=f"Advance to x{plan.rollback_start_step:02d} [rank {dist_ctx.rank}]") if class_indices else []
    history_corrector = history_tcc if args.tcc_prior_trajectory == "trajectory_consistent" else None
    for class_idx in iterator:
        base_start = load_state(line_state.base_dir, class_idx, dist_ctx.device)
        run_start = load_state(line_state.run_dir, class_idx, dist_ctx.device)
        base_target = advance_state(
            args,
            diffusion,
            base_model,
            base_start,
            class_idx,
            line_state.current_step,
            plan.rollback_start_step,
            dist_ctx.device,
            tcc_corrector=None,
        )
        run_target = advance_state(
            args,
            diffusion,
            run_model,
            run_start,
            class_idx,
            line_state.current_step,
            plan.rollback_start_step,
            dist_ctx.device,
            tcc_corrector=history_corrector,
        )
        save_state(target_base_dir, class_idx, base_target)
        save_state(target_run_dir, class_idx, run_target)

    barrier(dist_ctx)
    ensure_state_files(target_base_dir, class_indices, stage_name=f"advanced base_x{plan.rollback_start_step:02d}")
    ensure_state_files(target_run_dir, class_indices, stage_name=f"advanced run_x{plan.rollback_start_step:02d}")
    return LineState(current_step=plan.rollback_start_step, base_dir=target_base_dir, run_dir=target_run_dir)


def collect_local_step_stats(
    args,
    diffusion,
    base_model,
    run_model,
    line_state: LineState,
    plan: TargetStepPlan,
    class_indices,
    dist_ctx: DistContext,
    num_layers: int,
    token_len: int,
    hidden_dim: int,
    history_tcc: TccCorrector,
):
    ensure_state_files(
        line_state.base_dir,
        class_indices,
        stage_name=f"pre-collect base_x{plan.rollback_start_step:02d} for target {plan.target_step:02d}",
    )
    ensure_state_files(
        line_state.run_dir,
        class_indices,
        stage_name=f"pre-collect run_x{plan.rollback_start_step:02d} for target {plan.target_step:02d}",
    )

    if class_indices:
        print(
            f"[rank {dist_ctx.rank}] collect layer/branch TCC stats at step {plan.target_step:02d} "
            f"using DiT and history-TCC accelerated lines from x{plan.rollback_start_step:02d}",
            flush=True,
        )

    local_class_count = len(class_indices)
    local_token_cond_a = (
        torch.empty((2, num_layers, local_class_count, hidden_dim), dtype=torch.float32, device="cpu")
        if use_tokenpool(args)
        else None
    )
    local_token_cond_b = (
        torch.empty((2, num_layers, local_class_count, hidden_dim), dtype=torch.float32, device="cpu")
        if use_tokenpool(args)
        else None
    )
    local_class_sum_a = (
        torch.zeros((2, num_layers, token_len, hidden_dim), dtype=torch.float32, device="cpu")
        if use_classpool(args)
        else None
    )
    local_class_sum_b = (
        torch.zeros((2, num_layers, token_len, hidden_dim), dtype=torch.float32, device="cpu")
        if use_classpool(args)
        else None
    )
    history_corrector = history_tcc if args.tcc_prior_trajectory == "trajectory_consistent" else None

    iterator = tqdm(class_indices, desc=f"Collect step {plan.target_step:02d} [rank {dist_ctx.rank}]") if class_indices else []
    for local_idx, class_idx in enumerate(iterator):
        base_start = load_state(line_state.base_dir, class_idx, dist_ctx.device)
        run_start = load_state(line_state.run_dir, class_idx, dist_ctx.device)

        base_end, base_pack = run_window_with_target_collect(
            args,
            diffusion,
            base_model,
            base_start,
            class_idx,
            plan.rollback_start_step,
            plan.target_step,
            dist_ctx.device,
            tcc_corrector=None,
            collect_target_stats=True,
        )
        _, run_pack = run_window_with_target_collect(
            args,
            diffusion,
            run_model,
            run_start,
            class_idx,
            plan.rollback_start_step,
            plan.target_step,
            dist_ctx.device,
            tcc_corrector=history_corrector,
            collect_target_stats=True,
        )
        save_state(plan.next_base_dir, class_idx, base_end)
        if local_token_cond_a is not None:
            local_token_cond_a[:, :, local_idx, :].copy_(tokenpooled_cond_mean(base_pack))
            local_token_cond_b[:, :, local_idx, :].copy_(tokenpooled_cond_mean(run_pack))
        if local_class_sum_a is not None:
            local_class_sum_a += classpooled_cond_tokens(base_pack)
            local_class_sum_b += classpooled_cond_tokens(run_pack)

    torch.save(
        {
            "class_indices": class_indices,
            "tokenpool_cond_a": local_token_cond_a,
            "tokenpool_cond_b": local_token_cond_b,
            "classpool_sum_a": local_class_sum_a,
            "classpool_sum_b": local_class_sum_b,
            "classpool_count": int(local_class_count),
        },
        plan.local_stats_path,
    )
    barrier(dist_ctx)
    ensure_state_files(plan.next_base_dir, class_indices, stage_name=f"post-collect base_x{plan.next_state_step:02d}")


def build_global_step_pack(
    args,
    paths: RuntimePaths,
    plan: TargetStepPlan,
    dist_ctx: DistContext,
    num_classes: int,
    num_layers: int,
    hidden_dim: int,
    device: torch.device,
):
    if not is_main_process(dist_ctx):
        barrier(dist_ctx)
        return

    cond_a_tokenpool = (
        torch.empty((2, num_layers, num_classes, hidden_dim), dtype=torch.float32, device="cpu")
        if use_tokenpool(args)
        else None
    )
    cond_b_tokenpool = (
        torch.empty((2, num_layers, num_classes, hidden_dim), dtype=torch.float32, device="cpu")
        if use_tokenpool(args)
        else None
    )
    cond_a_classpool_sum = None
    cond_b_classpool_sum = None
    cond_a_classpool = None
    cond_b_classpool = None
    classpool_count_total = 0
    for worker_rank in range(dist_ctx.world_size):
        shard = torch.load(paths.shard_path(plan.target_step, worker_rank), map_location="cpu", weights_only=False)
        shard_indices = [int(x) for x in shard["class_indices"]]
        if shard_indices:
            # Use index_copy_ for advanced-index writes. Direct copy_ on
            # cond_a[:, :, shard_indices, :] would write into a temporary tensor
            # and leave most of the gathered tensor uninitialized.
            gather_idx = torch.as_tensor(shard_indices, dtype=torch.long)
            if cond_a_tokenpool is not None:
                cond_a_tokenpool.index_copy_(2, gather_idx, shard["tokenpool_cond_a"])
                cond_b_tokenpool.index_copy_(2, gather_idx, shard["tokenpool_cond_b"])
        if use_classpool(args):
            if shard.get("classpool_sum_a", None) is None or shard.get("classpool_sum_b", None) is None:
                raise RuntimeError(
                    f"Missing classpool sums in shard for target step {plan.target_step:02d}, rank {worker_rank}."
                )
            if cond_a_classpool_sum is None:
                cond_a_classpool_sum = torch.zeros_like(shard["classpool_sum_a"], dtype=torch.float32, device="cpu")
                cond_b_classpool_sum = torch.zeros_like(shard["classpool_sum_b"], dtype=torch.float32, device="cpu")
            cond_a_classpool_sum += shard["classpool_sum_a"].to(dtype=torch.float32)
            cond_b_classpool_sum += shard["classpool_sum_b"].to(dtype=torch.float32)
            classpool_count_total += int(shard.get("classpool_count", len(shard_indices)))

    if use_classpool(args):
        denom = float(max(classpool_count_total, 1))
        cond_a_classpool = cond_a_classpool_sum / denom
        cond_b_classpool = cond_b_classpool_sum / denom

    # Build the small global TCC pack on CPU. The inputs are already gathered on
    # CPU, and keeping the Procrustes SVDs off CUDA avoids delayed CUDA/NCCL
    # failures during distributed collection teardown.
    pack_build_device = torch.device("cpu")
    step_pack = build_step_pack(
        args,
        cond_a_tokenpool,
        cond_b_tokenpool,
        cond_a_classpool,
        cond_b_classpool,
        device=pack_build_device,
    )
    step_pack["step"] = int(plan.target_step)
    torch.save(step_pack, os.path.join(paths.pack_dir, f"step_{plan.target_step:02d}.pt"))
    barrier(dist_ctx)


def replay_current_step(
    args,
    diffusion,
    base_model,
    run_model,
    line_state: LineState,
    plan: TargetStepPlan,
    class_indices,
    dist_ctx: DistContext,
    current_tcc: TccCorrector,
):
    step_corrector = current_tcc if args.tcc_prior_trajectory == "trajectory_consistent" else None
    if class_indices:
        replay_mode = "current TCC pack" if step_corrector is not None else "uncorrected cache line"
        print(
            f"[rank {dist_ctx.rank}] replay {replay_mode} from x{plan.rollback_start_step:02d} "
            f"through target step {plan.target_step:02d} -> x{plan.next_state_step:02d}",
            flush=True,
        )

    base_model.to("cpu")
    if dist_ctx.device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        iterator = tqdm(class_indices, desc=f"Replay step {plan.target_step:02d} [rank {dist_ctx.rank}]") if class_indices else []
        for class_idx in iterator:
            run_start = load_state(line_state.run_dir, class_idx, dist_ctx.device)
            run_end, _ = run_window_with_target_collect(
                args,
                diffusion,
                run_model,
                run_start,
                class_idx,
                plan.rollback_start_step,
                plan.target_step,
                dist_ctx.device,
                tcc_corrector=step_corrector,
                collect_target_stats=False,
            )
            save_state(plan.next_run_dir, class_idx, run_end)
    finally:
        base_model.to(dist_ctx.device)
        if dist_ctx.device.type == "cuda":
            torch.cuda.empty_cache()

    barrier(dist_ctx)
    ensure_state_files(plan.next_run_dir, class_indices, stage_name=f"post-replay run_x{plan.next_state_step:02d}")


def process_target_step(
    args,
    diffusion,
    base_model,
    run_model,
    line_state: LineState,
    paths: RuntimePaths,
    dist_ctx: DistContext,
    class_indices,
    target_step: int,
    refresh_steps,
    num_layers: int,
    token_len: int,
    hidden_dim: int,
    history_tcc: TccCorrector,
):
    plan = make_target_step_plan(paths, target_step, refresh_steps, dist_ctx)
    ensure_valid_target_step(line_state, plan, refresh_steps)

    line_state = advance_lines_to_rollback_start(
        args,
        diffusion,
        base_model,
        run_model,
        line_state,
        plan,
        class_indices,
        dist_ctx,
        history_tcc,
        paths,
    )

    collect_local_step_stats(
        args,
        diffusion,
        base_model,
        run_model,
        line_state,
        plan,
        class_indices,
        dist_ctx,
        num_layers,
        token_len,
        hidden_dim,
        history_tcc,
    )

    build_global_step_pack(
        args,
        paths,
        plan,
        dist_ctx,
        args.num_classes,
        num_layers,
        hidden_dim,
        dist_ctx.device,
    )

    current_tcc = make_tcc_corrector(args, paths.pack_dir, dist_ctx.device)
    replay_current_step(
        args,
        diffusion,
        base_model,
        run_model,
        line_state,
        plan,
        class_indices,
        dist_ctx,
        current_tcc,
    )

    return LineState(
        current_step=plan.next_state_step,
        base_dir=plan.next_base_dir,
        run_dir=plan.next_run_dir,
    )


def save_summary(paths: RuntimePaths, line_state: LineState, target_steps, dist_ctx: DistContext):
    summary = {
        "final_base_state_dir": line_state.base_dir,
        "final_run_state_dir": line_state.run_dir,
        "tcc_pack_dir": paths.pack_dir,
        "target_steps_desc": [int(x) for x in target_steps],
        "final_state_step": int(line_state.current_step),
        "world_size": int(dist_ctx.world_size),
    }
    with open(os.path.join(paths.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main(args):
    args.accelerate_method = normalize_accelerate_method(args.accelerate_method)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    target_steps = sorted(parse_range_or_list(args.target_steps), reverse=True)
    validate_args(args, target_steps)

    refresh_steps = refresh_steps_for_args(args)

    dist_ctx = setup_distributed(args)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    base_model = build_model(args, diffusion, mode="base", device=dist_ctx.device)
    run_model = build_model(args, diffusion, mode="run", device=dist_ctx.device)
    class_indices = shard_class_indices(args.num_classes, dist_ctx)

    paths = build_runtime_paths(args.out_dir)
    if is_main_process(dist_ctx):
        save_meta(args, paths.out_dir, target_steps, refresh_steps)
    barrier(dist_ctx)

    line_state = initialize_line_state(args, paths, class_indices, dist_ctx)
    history_tcc = make_tcc_corrector(args, paths.pack_dir, dist_ctx.device)

    num_layers = len(run_model.blocks)
    hidden_dim = int(run_model.pos_embed.shape[-1])
    token_len = int(run_model.pos_embed.shape[1])

    for target_step in target_steps:
        line_state = process_target_step(
            args,
            diffusion,
            base_model,
            run_model,
            line_state,
            paths,
            dist_ctx,
            class_indices,
            target_step,
            refresh_steps,
            num_layers,
            token_len,
            hidden_dim,
            history_tcc,
        )

    if is_main_process(dist_ctx):
        save_summary(paths, line_state, target_steps, dist_ctx)
    barrier(dist_ctx)
    if dist_ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build online TCC packs by sequential replay on accelerated DiT lines.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Total number of latents to collect per class. --batch-size controls the per-forward micro-batch.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--target-steps", type=str, required=True, help='e.g. "18,16,14,12"')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--accelerate-method",
        type=str,
        default="fora",
        choices=["fora", "l2c"],
        help="Accelerated run line used as B. Use l2c for Learn-to-Cache or fora for FORA.",
    )
    parser.add_argument("--fora-interval", type=float, default=2.0)
    parser.add_argument("--path", type=str, default=None, help="Router checkpoint for --accelerate-method l2c.")
    parser.add_argument("--thres", type=float, default=0.1, help="Router threshold for --accelerate-method l2c.")
    parser.add_argument("--tcc-alpha", dest="tcc_alpha", metavar="ALPHA", type=float, default=0.25)
    parser.add_argument("--tcc-window", dest="tcc_window", metavar="WINDOW", type=str, default="12,18")
    parser.add_argument(
        "--tcc-prior-trajectory",
        dest="tcc_prior_trajectory",
        type=str,
        default="trajectory_consistent",
        choices=["trajectory_consistent", "one_shot"],
        help="Collect priors along the corrected TCC trajectory, or once from the uncorrected cache trajectory for the one-shot ablation.",
    )
    parser.add_argument(
        "--tcc-prior-pool-mode",
        dest="tcc_prior_pool_mode",
        type=str,
        default="tokenpool",
        choices=["tokenpool", "classpool", "mixed"],
        help="Prior aggregation mode used to build each online TCC pack.",
    )
    parser.add_argument(
        "--tcc-pack-variant",
        dest="tcc_pack_variant",
        type=str,
        default="full",
        metavar="{full,shift_only,scale_shift}",
        help=(
            "Linear correction stored in the TCC pack: "
            "full = shift + scale + Procrustes rotation; "
            "shift_only = shift without scale/rotation; "
            "scale_shift = shift + scalar scale without rotation."
        ),
    )
    parser.add_argument(
        "--tcc-prior-mix-classpool-ratio",
        dest="tcc_prior_mix_classpool_ratio",
        type=float,
        default=0.5,
        metavar="RATIO",
        help="When --tcc-prior-pool-mode=mixed, blend classpool correction with this ratio and tokenpool with (1-ratio).",
    )
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--init-states",
        action="store_true",
        help="Force regeneration of the staged latent states at the initial start step.",
    )
    parser.add_argument("--tf32", dest="tf32", action="store_true")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)
    main(parser.parse_args())
