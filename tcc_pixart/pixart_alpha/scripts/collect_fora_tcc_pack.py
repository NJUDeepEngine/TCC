import argparse
import os
import random
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


def load_prompts(path, limit, seed, selection):
    with open(path, "r") as f:
        rows = [(idx, line.strip()) for idx, line in enumerate(f) if line.strip()]
    if limit > 0 and limit < len(rows):
        if selection == "first":
            rows = rows[:limit]
        elif selection == "random":
            rng = random.Random(int(seed))
            rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), int(limit)))]
        else:
            raise ValueError(f"unsupported prompt selection: {selection}")
    return rows


def write_prompt_manifest(output_dir, prompt_rows):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "selected_prompts.txt"), "w") as f_prompt, open(
        os.path.join(output_dir, "selected_prompt_indices.txt"), "w"
    ) as f_index:
        for source_idx, prompt in prompt_rows:
            f_prompt.write(f"{prompt}\n")
            f_index.write(f"{source_idx}\n")


def prepare_batch(prompts, image_size, latent_size, base_ratios, device):
    clean = [prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip() for prompt in prompts]
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(len(clean), 1)
    ar = torch.tensor([[1.0]], dtype=torch.float, device=device).repeat(len(clean), 1)
    return clean, hw, ar, latent_size, latent_size


def make_latents(n, latent_h, latent_w, seed, prompt_indices, device):
    rows = []
    for prompt_idx in prompt_indices:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) * 1_000_003 + int(prompt_idx) * 97)
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


def shard_rows(rows, rank, world_size):
    return rows if world_size <= 1 else rows[rank::world_size]


def gather_stats(stats, distributed, rank, world_size, device):
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


def global_valid_mask(valid_mask, distributed, device):
    valid = valid_mask.to(device=device, dtype=torch.long)
    if distributed:
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    return valid.to(dtype=torch.bool).cpu()


@torch.inference_mode()
def run_dpm(args, model, t5, batch_rows, collector, force_full, history_steps, device, base_ratios):
    prompt_indices = [source_idx for source_idx, _ in batch_rows]
    prompts = [prompt for _, prompt in batch_rows]
    clean, hw, ar, latent_h, latent_w = prepare_batch(
        prompts, args.image_size, args.image_size // 8, base_ratios, device
    )
    caption_embs, emb_masks = t5.get_text_embeddings(clean)
    caption_embs = caption_embs.float()[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(len(clean), 1, 1)[:, None]
    z = make_latents(len(clean), latent_h, latent_w, args.seed, prompt_indices, device)
    tcc_corrector = None
    if not force_full and history_steps:
        tcc_corrector = TokenPoolTccCorrector(
            tcc_dir=args.output_dir,
            branches=BRANCHES_PIXART,
            alpha=args.tcc_alpha,
            target_steps=history_steps,
            device=device,
            cache_only=True,
            stale_only=args.stale_only,
        )
    model_kwargs = {
        "data_info": {"img_hw": hw, "aspect_ratio": ar},
        "mask": emb_masks,
        "cache_type": args.cache_type,
        "fresh_ratio": 0.0,
        "fresh_threshold": args.fresh_threshold,
        "force_fresh": "global",
        "ratio_scheduler": "constant",
        "soft_fresh_weight": 0.0,
        "cache_mode": "fora",
        "strict_force_fresh": True,
        "tcc_collector": collector,
        "tcc_corrector": tcc_corrector,
        "tcc_force_full": bool(force_full),
    }
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
        iterator = tqdm(iterator, desc=f"strict FORA collect step {target_step}")
    for start in iterator:
        batch_rows = prompt_rows[start : start + args.batch_size]
        run_dpm(args, model, t5, batch_rows, a_collector, True, history_steps, device, base_ratios)
        run_dpm(args, model, t5, batch_rows, b_collector, False, history_steps, device, base_ratios)

    has_any = torch.tensor([1 if b_collector.has_any() else 0], dtype=torch.long, device=device)
    if distributed:
        dist.all_reduce(has_any, op=dist.ReduceOp.SUM)
    if int(has_any.item()) == 0:
        if rank == 0:
            print(f"[skip] step {target_step} is not a strict FORA cache step; no pack written.")
        return False

    a_tensor, a_valid = a_collector.tensor_allow_missing()
    b_tensor, b_valid = b_collector.tensor_allow_missing()
    valid_mask = global_valid_mask(a_valid & b_valid, distributed, device)
    if int(valid_mask.sum().item()) == 0:
        if rank == 0:
            print(f"[skip] step {target_step} produced no complete TCC stats.")
        return False
    a_stats = gather_stats(a_tensor, distributed, rank, world_size, device)
    b_stats = gather_stats(b_tensor, distributed, rank, world_size, device)
    if rank != 0:
        return True
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


def main():
    parser = argparse.ArgumentParser(
        description="Collect TCC packs on strict PixArt FORA cache steps. "
        "Use --fresh-threshold to choose N and pass only actual cache-step indices in --targets."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--txt-file", required=True)
    parser.add_argument("--num-prompts", type=int, default=5000)
    parser.add_argument("--prompt-selection", choices=["first", "random"], default="first")
    parser.add_argument("--targets", default="1,2,4,5")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512, 1024])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--t5-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--fresh-threshold", type=int, default=3)
    parser.add_argument("--cache-type", type=str, default="attention")
    parser.add_argument("--tcc-alpha", type=float, default=0.5)
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
    t5_device = str(device) if device.type == "cuda" else "cpu"
    t5 = T5Embedder(device=t5_device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")

    prompt_rows_all = load_prompts(args.txt_file, args.num_prompts, args.seed, args.prompt_selection)
    if rank == 0:
        write_prompt_manifest(args.output_dir, prompt_rows_all)
        print(f"[collect] strict FORA n={args.fresh_threshold}, targets={args.targets}, prompts={len(prompt_rows_all)}")
    if distributed:
        dist.barrier()
    prompt_rows = shard_rows(prompt_rows_all, rank, world_size)
    targets = parse_int_list(args.targets)
    history = []
    for target in targets:
        wrote_pack = collect_target(
            args, target, history, prompt_rows, model, t5, device, base_ratios, distributed, rank, world_size
        )
        if wrote_pack:
            history.append(int(target))
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
