import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.tcc_online import BRANCHES_PIXART, TokenPoolTccCorrector, parse_int_list  # noqa: E402
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


def write_prompt_manifest(sample_dir, prompt_rows):
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "selected_prompts.txt"), "w") as f_prompt, open(
        os.path.join(sample_dir, "selected_prompt_indices.txt"), "w"
    ) as f_index:
        for source_idx, prompt in prompt_rows:
            f_prompt.write(f"{prompt}\n")
            f_index.write(f"{source_idx}\n")


def prepare_batch(prompts, image_size, latent_size, base_ratios, device):
    clean = [prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip() for prompt in prompts]
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(len(clean), 1)
    ar = torch.tensor([[1.0]], dtype=torch.float, device=device).repeat(len(clean), 1)
    return clean, hw, ar, latent_size, latent_size


def set_env(seed, rank, device):
    global_seed = int(seed) + int(rank)
    torch.manual_seed(global_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(global_seed)
    torch.set_grad_enabled(False)


def discover_tcc_targets(tcc_dir):
    if not tcc_dir:
        return []
    targets = []
    for name in os.listdir(tcc_dir):
        match = re.fullmatch(r"step_(\d+)\.pt", name)
        if match:
            targets.append(int(match.group(1)))
    return sorted(targets)


def resolve_tcc_targets(args):
    if args.tcc_targets:
        return parse_int_list(args.tcc_targets)
    if args.tcc_window:
        lo, hi = parse_int_list(args.tcc_window)
        return list(range(int(lo), int(hi) + 1))
    return discover_tcc_targets(args.tcc_dir)


def create_npz_from_sample_folder(sample_dir, num, sample_ext):
    samples = []
    for i in tqdm(range(num), desc="build npz"):
        with Image.open(os.path.join(sample_dir, f"{i:06d}.{sample_ext}")) as sample_pil:
            samples.append(np.asarray(sample_pil.convert("RGB"), dtype=np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"saved npz: {npz_path} shape={samples.shape}")
    return npz_path


@torch.inference_mode()
def sample_batch(args, model, vae, t5, prompts, device, base_ratios, tcc_corrector):
    clean, hw, ar, latent_h, latent_w = prepare_batch(
        prompts, args.image_size, args.image_size // 8, base_ratios, device
    )
    caption_embs, emb_masks = t5.get_text_embeddings(clean)
    caption_embs = caption_embs.float()[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(len(clean), 1, 1)[:, None]
    z = torch.randn(len(clean), 4, latent_h, latent_w, device=device)
    profile = {}
    model_kwargs = {
        "data_info": {"img_hw": hw, "aspect_ratio": ar},
        "mask": emb_masks,
        "cache_type": args.cache_type,
        "fresh_ratio": args.fresh_ratio,
        "fresh_threshold": args.fresh_threshold,
        "force_fresh": args.force_fresh,
        "ratio_scheduler": args.ratio_scheduler,
        "soft_fresh_weight": args.soft_fresh_weight,
        "use_toca": not args.native_pixart,
        "strict_force_fresh": args.strict_force_fresh,
        "cache_mode": args.cache_mode,
        "tcc_corrector": tcc_corrector,
        "test_FLOPs": args.profile or args.test_FLOPs,
        "profile": profile,
    }
    dpm_solver = DPMS(
        model.forward_with_dpmsolver,
        condition=caption_embs,
        uncondition=null_y,
        cfg_scale=args.cfg_scale,
        model_kwargs=model_kwargs,
    )
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
    samples = dpm_solver.sample(
        z,
        steps=args.num_sampling_steps,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        model_kwargs=model_kwargs,
        rank=None,
    )
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize(device)
        latency_ms = start.elapsed_time(end)
    else:
        latency_ms = float("nan")
    samples = vae.decode(samples / 0.18215).sample
    samples = samples.add(1).div(2).clamp(0, 1).mul(255).add_(0.5).clamp_(0, 255)
    return samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy(), latency_ms, profile


def main():
    parser = argparse.ArgumentParser(description="PixArt sampling, TCC application, and optional profiling.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--txt-file", required=True)
    parser.add_argument("--num-samples", type=int, default=30000)
    parser.add_argument("--prompt-selection", choices=["first", "random"], default="first")
    parser.add_argument("--batch-size", type=int, default=100)
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
    parser.add_argument("--native-pixart", action="store_true")
    parser.add_argument("--cache-mode", choices=["toca", "fora"], default="toca")
    parser.add_argument("--strict-force-fresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tcc-dir", type=str, default=None)
    parser.add_argument("--tcc-alpha", type=float, default=1.0)
    parser.add_argument("--tcc-targets", type=str, default=None)
    parser.add_argument("--tcc-window", type=str, default=None)
    parser.add_argument("--tcc-stale-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-ext", choices=["png", "jpg"], default="jpg")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--test-FLOPs", action="store_true")
    parser.add_argument("--profile-json", default=None)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    distributed = bool(args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1)
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_env(args.seed, rank, device)
    torch.backends.cuda.matmul.allow_tf32 = True
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

    vae = AutoencoderKL.from_pretrained(args.tokenizer_path).to(device)
    t5_device = str(device) if device.type == "cuda" else "cpu"
    t5 = T5Embedder(device=t5_device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    prompt_rows = load_prompts(args.txt_file, args.num_samples, args.seed, args.prompt_selection)
    sample_count = len(prompt_rows)
    os.makedirs(args.sample_dir, exist_ok=True)
    if rank == 0:
        write_prompt_manifest(args.sample_dir, prompt_rows)
    if distributed:
        dist.barrier()

    if args.tcc_dir:
        tcc_targets = resolve_tcc_targets(args)
        if rank == 0:
            print(f"using online TCC dir={args.tcc_dir}, alpha={args.tcc_alpha}, targets={tcc_targets}")
        tcc_corrector = TokenPoolTccCorrector(
            tcc_dir=args.tcc_dir,
            branches=BRANCHES_PIXART,
            alpha=args.tcc_alpha,
            target_steps=tcc_targets,
            device=device,
            cache_only=True,
            stale_only=args.tcc_stale_only,
        )
    else:
        tcc_corrector = None

    local_indices = list(range(rank, sample_count, world_size))
    iterator = range(0, len(local_indices), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"sample rank{rank}/{world_size}")
    saved = 0
    latencies_ms = []
    tflops = []
    model_tflops = []
    tcc_tflops = []
    for local_start in iterator:
        batch_indices = local_indices[local_start : local_start + args.batch_size]
        batch = [prompt_rows[i][1] for i in batch_indices]
        samples, latency_ms, profile = sample_batch(args, model, vae, t5, batch, device, base_ratios, tcc_corrector)
        latencies_ms.append(latency_ms)
        tflops.append(float(profile.get("tflops", 0.0)))
        model_tflops.append(float(profile.get("model_tflops", profile.get("tflops", 0.0))))
        tcc_tflops.append(float(profile.get("tcc_tflops", 0.0)))
        if not args.profile:
            for idx, sample in zip(batch_indices, samples):
                Image.fromarray(sample).save(os.path.join(args.sample_dir, f"{idx:06d}.{args.sample_ext}"))
                saved += 1

    if tcc_corrector is not None:
        print(
            f"online TCC applied calls={tcc_corrector.apply_calls}, "
            f"corrected_elements={tcc_corrector.corrected_elements}, "
            f"calls_by_step={dict(sorted(tcc_corrector.calls_by_step.items()))}"
        )

    profile_summary = {
        "native_pixart": bool(args.native_pixart),
        "use_toca": not bool(args.native_pixart),
        "num_sampling_steps": int(args.num_sampling_steps),
        "fresh_threshold": int(args.fresh_threshold),
        "fresh_ratio": float(args.fresh_ratio),
        "batch_size": int(args.batch_size),
        "local_batches": len(latencies_ms),
        "latency_ms_mean_per_batch": float(np.nanmean(latencies_ms)) if latencies_ms else None,
        "latency_ms_mean_per_image": float(np.nanmean(latencies_ms) / args.batch_size) if latencies_ms else None,
        "tflops_mean_per_batch": float(np.mean(tflops)) if tflops else None,
        "tflops_mean_per_image": float(np.mean(tflops) / args.batch_size) if tflops else None,
        "model_tflops_mean_per_batch": float(np.mean(model_tflops)) if model_tflops else None,
        "model_tflops_mean_per_image": float(np.mean(model_tflops) / args.batch_size) if model_tflops else None,
        "tcc_tflops_mean_per_batch": float(np.mean(tcc_tflops)) if tcc_tflops else None,
        "tcc_tflops_mean_per_image": float(np.mean(tcc_tflops) / args.batch_size) if tcc_tflops else None,
    }
    print(json.dumps(profile_summary, indent=2, sort_keys=True))
    if rank == 0 and args.profile_json:
        os.makedirs(os.path.dirname(args.profile_json), exist_ok=True)
        with open(args.profile_json, "w") as f:
            json.dump(profile_summary, f, indent=2, sort_keys=True)

    if distributed:
        dist.barrier()
    if rank == 0 and args.make_npz and not args.profile:
        create_npz_from_sample_folder(args.sample_dir, sample_count, args.sample_ext)
    total_target = int(math.ceil(sample_count / args.batch_size) * args.batch_size)
    print(f"[rank {rank}] done: saved {saved}/{sample_count} samples in {args.sample_dir}; padded target={total_target}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
