import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pack_collection.tcc_online import (  # noqa: E402
    BRANCHES_DIT,
    TokenPoolCollector,
    TokenPoolTccCorrector,
    build_tokenpool_pack,
    parse_int_list,
)
from diffusion import create_diffusion  # noqa: E402
from download import find_model  # noqa: E402
from models import DiT_models  # noqa: E402


def distributed_is_available():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not distributed_is_available():
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return int(rank) == 0


def make_latents(labels, *, samples_per_label, latent_size, seed, device, sample_offset=0):
    chunks = []
    for label in labels:
        for sample_idx in range(samples_per_label):
            gen = torch.Generator(device="cpu")
            global_sample_idx = int(sample_offset) + int(sample_idx)
            gen.manual_seed(int(seed) * 1_000_003 + int(label) * 97 + global_sample_idx)
            chunks.append(torch.randn((1, 4, latent_size, latent_size), generator=gen, dtype=torch.float32))
    return torch.cat(chunks, dim=0).to(device)


def make_labels(labels, *, samples_per_label, device):
    rows = []
    for label in labels:
        rows.extend([int(label)] * int(samples_per_label))
    return torch.tensor(rows, device=device, dtype=torch.long)


@torch.inference_mode()
def run_to_target(
    *,
    args,
    model,
    diffusion,
    x_half,
    y_cond,
    target_step,
    device,
    collector=None,
    corrector=None,
    force_full=False,
):
    n = x_half.shape[0]
    x = torch.cat([x_half, x_half], dim=0)
    y_null = torch.full((n,), args.num_classes, device=device, dtype=torch.long)
    y = torch.cat([y_cond, y_null], dim=0)
    model_kwargs = {
        "y": y,
        "cfg_scale": args.cfg_scale,
        "cache_type": args.cache_type,
        "fresh_ratio": args.fresh_ratio,
        "fresh_threshold": args.fresh_threshold,
        "force_fresh": args.force_fresh,
        "ratio_scheduler": args.ratio_scheduler,
        "soft_fresh_weight": args.soft_fresh_weight,
        "test_FLOPs": False,
        "tcc_collector": collector,
        "tcc_corrector": corrector,
        "tcc_force_full": bool(force_full),
    }

    from cache_functions import cache_init

    cache_dic, current = cache_init(model_kwargs=model_kwargs, num_steps=args.num_sampling_steps)
    for step in range(args.num_sampling_steps - 1, int(target_step) - 1, -1):
        current["step"] = int(step)
        t = torch.full((x.shape[0],), int(step), device=device, dtype=torch.long)
        out = diffusion.ddim_sample(
            model.forward_with_cfg,
            x,
            t,
            current=current,
            cache_dic=cache_dic,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            eta=0.0,
        )
        x = out["sample"]
    return x


def gather_collector_tensor(local_tensor, rank, world_size):
    if world_size == 1:
        return local_tensor
    gathered = [None for _ in range(world_size)] if is_main_process(rank) else None
    dist.gather_object(local_tensor.cpu(), object_gather_list=gathered, dst=0)
    if not is_main_process(rank):
        return None
    return torch.cat(gathered, dim=2)


def collector_tensor_or_empty(collector):
    if collector.has_any():
        return collector.tensor_allow_missing()
    stats = torch.empty(
        (len(collector.branches), collector.num_layers, 0, collector.hidden_dim),
        dtype=torch.float32,
    )
    valid = torch.zeros((len(collector.branches), collector.num_layers), dtype=torch.bool)
    return stats, valid


def reduce_valid_mask(local_valid, device, world_size):
    valid = local_valid.to(device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    return valid.to(device="cpu", dtype=torch.bool)


def average_samples_per_label(stats, samples_per_label):
    samples_per_label = int(samples_per_label)
    if samples_per_label <= 1 or stats.shape[2] == 0:
        return stats
    if stats.shape[2] % samples_per_label != 0:
        raise RuntimeError(
            f"collector sample axis length {stats.shape[2]} is not divisible by "
            f"samples_per_label={samples_per_label}"
        )
    num_labels = stats.shape[2] // samples_per_label
    return stats.reshape(
        stats.shape[0],
        stats.shape[1],
        num_labels,
        samples_per_label,
        stats.shape[3],
    ).mean(dim=3)


def collect_target(args, target_step, history_steps, labels, model, diffusion, device, rank, world_size):
    latent_size = args.image_size // 8
    hidden_dim = int(args.hidden_dim)
    a_collector = TokenPoolCollector(
        target_step=target_step,
        branches=BRANCHES_DIT,
        num_layers=args.num_layers,
        hidden_dim=hidden_dim,
        cond_batch_size=args.batch_size,
        stale_only=args.stale_only,
    )
    b_collector = TokenPoolCollector(
        target_step=target_step,
        branches=BRANCHES_DIT,
        num_layers=args.num_layers,
        hidden_dim=hidden_dim,
        cond_batch_size=args.batch_size,
        stale_only=args.stale_only,
        require_cache_step=True,
    )
    history_tcc = TokenPoolTccCorrector(
        tcc_dir=args.output_dir,
        branches=BRANCHES_DIT,
        alpha=args.tcc_alpha,
        target_steps=history_steps,
        device=device,
        cache_only=True,
        stale_only=args.stale_only,
    )

    for label in tqdm(
        labels,
        desc=f"collect step {target_step}",
        disable=not is_main_process(rank),
    ):
        collected = 0
        while collected < int(args.samples_per_label):
            chunk_size = min(int(args.batch_size), int(args.samples_per_label) - collected)
            batch_labels = [int(label)]
            x0 = make_latents(
                batch_labels,
                samples_per_label=chunk_size,
                latent_size=latent_size,
                seed=args.seed,
                device=device,
                sample_offset=collected,
            )
            y = make_labels(batch_labels, samples_per_label=chunk_size, device=device)
            run_to_target(
                args=args,
                model=model,
                diffusion=diffusion,
                x_half=x0,
                y_cond=y,
                target_step=target_step,
                device=device,
                collector=a_collector,
                corrector=None,
                force_full=True,
            )
            run_to_target(
                args=args,
                model=model,
                diffusion=diffusion,
                x_half=x0,
                y_cond=y,
                target_step=target_step,
                device=device,
                collector=b_collector,
                corrector=history_tcc,
                force_full=False,
            )
            collected += chunk_size

    local_has_any = torch.tensor([int(b_collector.has_any())], device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(local_has_any, op=dist.ReduceOp.SUM)

    if int(local_has_any.item()) == 0:
        if is_main_process(rank):
            print(
                f"[skip] requested target step {target_step} did not hit a ToCa cache/reuse branch "
                "under the current ToCa schedule; no cache-only TCC pack is written."
            )
        return False

    a_stats_local, a_valid_local = collector_tensor_or_empty(a_collector)
    b_stats_local, b_valid_local = collector_tensor_or_empty(b_collector)
    a_stats_local = average_samples_per_label(a_stats_local, args.samples_per_label)
    b_stats_local = average_samples_per_label(b_stats_local, args.samples_per_label)
    valid_mask = reduce_valid_mask(a_valid_local & b_valid_local, device, world_size)
    a_stats = gather_collector_tensor(a_stats_local, rank, world_size)
    b_stats = gather_collector_tensor(b_stats_local, rank, world_size)

    if not is_main_process(rank):
        return True

    if b_stats is None or b_stats.shape[2] == 0:
        print(
            f"[skip] requested target step {target_step} did not hit a ToCa cache/reuse branch "
            "under the current ToCa schedule; no cache-only TCC pack is written."
        )
        return False

    pack = build_tokenpool_pack(
        a_stats,
        b_stats,
        branches=BRANCHES_DIT,
        target_step=target_step,
        device=device,
        valid_mask=valid_mask,
    )
    pack["samples_per_label"] = int(args.samples_per_label)
    pack["sample_axis_name"] = "imagenet_label_mean"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"step_{int(target_step):02d}.pt")
    torch.save(pack, out_path)
    print(f"saved {out_path}")
    return True


def resolve_targets(args):
    if args.targets:
        return parse_int_list(args.targets)
    lo, hi = parse_int_list(args.target_window)
    return list(range(int(hi), int(lo) - 1, -1))


def write_label_manifest(output_dir, labels):
    os.makedirs(output_dir, exist_ok=True)
    label_path = os.path.join(output_dir, "selected_labels.txt")
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{int(label)}\n")
    print(f"wrote label manifest: {label_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay-based online tokenpool TCC collection for DiT-ToCa.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-window", default="30,48")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--debug-targets", default=None)
    parser.add_argument("--num-representatives", type=int, default=1000)
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=50,
        help="Total number of representative latents collected for each ImageNet label.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Actual per-forward conditional batch size used while collecting one label's representatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--fresh-threshold", type=int, default=2, help="ToCa cache interval N.")
    parser.add_argument("--fresh-ratio", type=float, default=0.07)
    parser.add_argument("--cache-type", type=str, choices=["attention"], default="attention")
    parser.add_argument("--ratio-scheduler", type=str, choices=["ToCa-ddim50"], default="ToCa-ddim50")
    parser.add_argument("--force-fresh", type=str, choices=["global"], default="global")
    parser.add_argument("--soft-fresh-weight", type=float, default=0.25)
    parser.add_argument("--tcc-alpha", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1152)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--stale-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model.load_state_dict(find_model(ckpt_path))
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    labels = list(range(min(args.num_representatives, args.num_classes)))
    if is_main_process(rank):
        print(f"Starting collect rank={rank}, world_size={world_size}, local_rank={local_rank}.")
        write_label_manifest(args.output_dir, labels)
    targets = parse_int_list(args.debug_targets) if args.debug_targets else resolve_targets(args)
    rank_labels = labels[rank::world_size]
    history = []
    try:
        for target in targets:
            wrote_pack = collect_target(args, target, history, rank_labels, model, diffusion, device, rank, world_size)
            if world_size > 1:
                dist.barrier()
            if wrote_pack:
                history.append(int(target))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
