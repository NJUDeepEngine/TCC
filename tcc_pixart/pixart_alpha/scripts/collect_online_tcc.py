import argparse
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.tcc_online import (  # noqa: E402
    BRANCHES_PIXART,
    TokenPoolCollector,
    TokenPoolTccCorrector,
    build_tokenpool_pack,
    parse_int_list,
)
from diffusion import DPMS  # noqa: E402
from diffusion.data.datasets import ASPECT_RATIO_256_TEST, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST  # noqa: E402
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2  # noqa: E402
from diffusion.model.t5 import T5Embedder  # noqa: E402
from diffusion.model.utils import prepare_prompt_ar  # noqa: E402
from tools.download import find_model  # noqa: E402


def load_prompts(path, *, limit, seed, selection):
    with open(path, "r") as f:
        rows = [(idx, line.strip()) for idx, line in enumerate(f) if line.strip()]
    if limit is not None and limit > 0 and limit < len(rows):
        rng = random.Random(int(seed))
        if selection == "random":
            selected = sorted(rng.sample(range(len(rows)), int(limit)))
            rows = [rows[i] for i in selected]
        elif selection == "first":
            rows = rows[: int(limit)]
        else:
            raise ValueError(f"unsupported prompt selection mode: {selection}")
    return rows


def write_prompt_manifest(output_dir, prompt_rows):
    os.makedirs(output_dir, exist_ok=True)
    prompt_path = os.path.join(output_dir, "selected_prompts.txt")
    index_path = os.path.join(output_dir, "selected_prompt_indices.txt")
    with open(prompt_path, "w") as f_prompt, open(index_path, "w") as f_index:
        for source_idx, prompt in prompt_rows:
            f_prompt.write(f"{prompt}\n")
            f_index.write(f"{source_idx}\n")
    print(f"wrote prompt manifest: {prompt_path}")


def prepare_batch(prompts, *, image_size, latent_size, base_ratios, device):
    clean = []
    for prompt in prompts:
        clean.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(len(clean), 1)
    ar = torch.tensor([[1.0]], dtype=torch.float, device=device).repeat(len(clean), 1)
    return clean, hw, ar, latent_size, latent_size


def make_latents(n, *, latent_h, latent_w, seed, prompt_offset, device, prompt_indices=None):
    rows = []
    for i in range(n):
        prompt_idx = int(prompt_indices[i]) if prompt_indices is not None else int(prompt_offset + i)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * 1_000_003 + prompt_idx * 97)
        rows.append(torch.randn((1, 4, latent_h, latent_w), generator=gen, dtype=torch.float32))
    return torch.cat(rows, dim=0).to(device)


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def rank_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def shard_rows(rows, rank, world_size):
    if world_size <= 1:
        return rows
    return rows[rank::world_size]


def gather_stats(stats, *, distributed, rank, world_size, device):
    if not distributed:
        return stats

    local_len = torch.tensor([stats.shape[2]], dtype=torch.long, device=device)
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths = [int(x.item()) for x in lengths]
    max_len = max(lengths)

    local = stats.to(device=device)
    if local.shape[2] < max_len:
        pad_shape = (*local.shape[:2], max_len - local.shape[2], local.shape[3])
        local = torch.cat([local, torch.zeros(pad_shape, dtype=local.dtype, device=device)], dim=2)

    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if rank != 0:
        return None

    chunks = [tensor[:, :, :length, :].cpu() for tensor, length in zip(gathered, lengths) if length > 0]
    return torch.cat(chunks, dim=2)


def global_valid_mask(valid_mask, *, distributed, device):
    valid = valid_mask.to(device=device, dtype=torch.long)
    if distributed:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    return valid.to(dtype=torch.bool).cpu()


@torch.inference_mode()
def run_dpm(
    *,
    args,
    model,
    t5,
    prompts,
    prompt_indices,
    prompt_offset,
    target_step,
    history_steps,
    collector,
    force_full,
    device,
    base_ratios,
):
    clean, hw, ar, latent_h, latent_w = prepare_batch(
        prompts,
        image_size=args.image_size,
        latent_size=args.image_size // 8,
        base_ratios=base_ratios,
        device=device,
    )
    caption_embs, emb_masks = t5.get_text_embeddings(clean)
    caption_embs = caption_embs.float()[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(len(clean), 1, 1)[:, None]
    z = make_latents(
        len(clean),
        latent_h=latent_h,
        latent_w=latent_w,
        seed=args.seed,
        prompt_offset=prompt_offset,
        device=device,
        prompt_indices=prompt_indices,
    )
    model_kwargs = {
        "data_info": {"img_hw": hw, "aspect_ratio": ar},
        "mask": emb_masks,
        "cache_type": args.cache_type,
        "fresh_ratio": args.fresh_ratio,
        "fresh_threshold": args.fresh_threshold,
        "force_fresh": args.force_fresh,
        "ratio_scheduler": args.ratio_scheduler,
        "soft_fresh_weight": args.soft_fresh_weight,
        "tcc_collector": collector,
        "tcc_corrector": None,
        "tcc_force_full": bool(force_full),
    }
    if not force_full:
        model_kwargs["tcc_corrector"] = TokenPoolTccCorrector(
            tcc_dir=args.output_dir,
            branches=BRANCHES_PIXART,
            alpha=args.tcc_alpha,
            target_steps=history_steps,
            device=device,
            cache_only=True,
            stale_only=args.stale_only,
        )

    dpm_solver = DPMS(
        model.forward_with_dpmsolver,
        condition=caption_embs,
        uncondition=null_y,
        cfg_scale=args.cfg_scale,
        model_kwargs=model_kwargs,
    )
    dpm_solver.sample(
        z,
        steps=args.num_sampling_steps,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        model_kwargs=model_kwargs,
        rank=None,
    )


def collect_target(args, target_step, history_steps, prompt_rows, model, t5, device, base_ratios, distributed, rank, world_size):
    a_collector = TokenPoolCollector(
        target_step=target_step,
        branches=BRANCHES_PIXART,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        cond_batch_size=args.batch_size,
        stale_only=args.stale_only,
    )
    b_collector = TokenPoolCollector(
        target_step=target_step,
        branches=BRANCHES_PIXART,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        cond_batch_size=args.batch_size,
        stale_only=args.stale_only,
        require_cache_step=True,
    )
    iterator = range(0, len(prompt_rows), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"collect step {target_step}")
    for start in iterator:
        batch_rows = prompt_rows[start : start + args.batch_size]
        batch_indices = [source_idx for source_idx, _ in batch_rows]
        batch = [prompt for _, prompt in batch_rows]
        run_dpm(
            args=args,
            model=model,
            t5=t5,
            prompts=batch,
            prompt_indices=batch_indices,
            prompt_offset=batch_indices[0] if batch_indices else start,
            target_step=target_step,
            history_steps=history_steps,
            collector=a_collector,
            force_full=True,
            device=device,
            base_ratios=base_ratios,
        )
        run_dpm(
            args=args,
            model=model,
            t5=t5,
            prompts=batch,
            prompt_indices=batch_indices,
            prompt_offset=batch_indices[0] if batch_indices else start,
            target_step=target_step,
            history_steps=history_steps,
            collector=b_collector,
            force_full=False,
            device=device,
            base_ratios=base_ratios,
        )

    has_any = torch.tensor([1 if b_collector.has_any() else 0], dtype=torch.long, device=device)
    if distributed:
        dist.all_reduce(has_any, op=dist.ReduceOp.SUM)
    if int(has_any.item()) == 0:
        rank_print(
            rank,
            f"[skip] requested target step {target_step} did not hit a ToCa cache/reuse branch "
            "under the current ToCa schedule; no cache-only TCC pack is written.",
        )
        return False

    a_tensor, a_valid = a_collector.tensor_allow_missing()
    b_tensor, b_valid = b_collector.tensor_allow_missing()
    valid_mask = global_valid_mask(a_valid & b_valid, distributed=distributed, device=device)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        rank_print(rank, f"[skip] target step {target_step} produced no complete branch/layer TCC stats.")
        return False

    a_stats = gather_stats(a_tensor, distributed=distributed, rank=rank, world_size=world_size, device=device)
    b_stats = gather_stats(b_tensor, distributed=distributed, rank=rank, world_size=world_size, device=device)
    if rank != 0:
        return True

    if valid_count < valid_mask.numel():
        missing = valid_mask.numel() - valid_count
        print(f"[warn] target step {target_step}: {missing}/{valid_mask.numel()} branch/layer entries missing; they will be identity no-op.")

    pack = build_tokenpool_pack(
        a_stats,
        b_stats,
        branches=BRANCHES_PIXART,
        target_step=target_step,
        device=device,
        valid_mask=valid_mask,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"step_{int(target_step):02d}.pt")
    torch.save(pack, out_path)
    print(f"saved {out_path}")
    return True


def resolve_targets(args):
    if args.targets:
        return parse_int_list(args.targets)
    lo, hi = parse_int_list(args.target_window)
    return list(range(int(lo), int(hi) + 1))


def main():
    parser = argparse.ArgumentParser(description="Replay-based online tokenpool TCC collection for PixArt-ToCa.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--txt-file", required=True)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--prompt-selection", choices=["first", "random"], default="first")
    parser.add_argument("--target-window", default="12,18")
    parser.add_argument("--targets", default=None)
    parser.add_argument("--debug-targets", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512, 1024])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--t5-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--fresh-threshold", type=int, default=3)
    parser.add_argument("--fresh-ratio", type=float, default=0.30)
    parser.add_argument("--cache-type", type=str, default="attention")
    parser.add_argument("--ratio-scheduler", type=str, default="ToCa")
    parser.add_argument("--force-fresh", type=str, default="global")
    parser.add_argument("--soft-fresh-weight", type=float, default=0.25)
    parser.add_argument("--tcc-alpha", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1152)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--stale-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed) + int(rank))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(args.seed) + int(rank))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    latent_size = args.image_size // 8
    lewei_scale = {256: 1, 512: 1, 1024: 2}
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if args.image_size in [256, 512]:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    state_dict = find_model(args.model_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict.pop("pos_embed", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(weight_dtype)
    t5 = T5Embedder(device="cuda" if device.type == "cuda" else "cpu", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")

    prompt_rows_all = load_prompts(args.txt_file, limit=args.num_prompts, seed=args.seed, selection=args.prompt_selection)
    if rank == 0:
        write_prompt_manifest(args.output_dir, prompt_rows_all)
    if distributed:
        dist.barrier()
    prompt_rows = shard_rows(prompt_rows_all, rank, world_size)
    rank_print(rank, f"[collect-ddp] world_size={world_size}, total_prompts={len(prompt_rows_all)}")
    print(f"[rank {rank}] collect prompts={len(prompt_rows)}")
    targets = parse_int_list(args.debug_targets) if args.debug_targets else resolve_targets(args)
    history = []
    for target in targets:
        wrote_pack = collect_target(args, target, history, prompt_rows, model, t5, device, base_ratios, distributed, rank, world_size)
        if wrote_pack:
            history.append(int(target))
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
