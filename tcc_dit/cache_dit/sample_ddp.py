# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from pathlib import Path
from typing import Tuple
from tcc_correct import TccCorrector, parse_tcc_window

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def resolve_vae_source(vae_name: str, vae_repo_path: str | None = None) -> Tuple[str, bool]:
    if vae_repo_path:
        return vae_repo_path, Path(vae_repo_path).expanduser().exists()

    env_vae_dir = os.environ.get("TCC_VAE_DIR")
    if env_vae_dir:
        vae_path = Path(env_vae_dir).expanduser()
        if vae_path.is_dir():
            return str(vae_path), True
        print(f"[WARN] TCC_VAE_DIR does not exist, falling back: {vae_path}")

    cache_roots = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        env_value = os.environ.get(env_name)
        if env_value:
            cache_roots.append(Path(env_value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    local_candidates = [
        cache_root / f"models--stabilityai--sd-vae-ft-{vae_name}" / "snapshots"
        for cache_root in cache_roots
    ]
    for candidate in local_candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "config.json").is_file():
            return str(candidate), True
        snapshots = sorted([p for p in candidate.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1]), True
    return f"stabilityai/sd-vae-ft-{vae_name}", False


def normalize_accelerate_method(method: str | None) -> str | None:
    return method


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    diffusion = create_diffusion(str(args.num_sampling_steps))
    args.accelerate_method = normalize_accelerate_method(args.accelerate_method)

    l2c_enabled = args.l2c_enable or args.accelerate_method == "l2c"
    fora_enabled = args.accelerate_method == "fora"
    if l2c_enabled and args.accelerate_method is None:
        args.accelerate_method = "l2c"
    if args.accelerate_method not in (None, "fora", "l2c"):
        raise ValueError("--accelerate-method supports only fora or l2c in this release")
    if args.tcc_enable and not (l2c_enabled or fora_enabled):
        raise ValueError("TCC currently requires a cache-enabled mode: l2c or fora")
    if l2c_enabled and args.path is None:
        raise ValueError("L2C requires --path to the trained router checkpoint")

    # Load model:
    latent_size = args.image_size // 8
    if args.accelerate_method in ("l2c", "fora"):
        from models.dynamic_models import DiT_models
    else:
        from models.models import DiT_models

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    if args.accelerate_method is not None:
        if args.accelerate_method == "l2c":
            model.load_ranking(args.path, args.num_sampling_steps, diffusion.timestep_map, args.thres)
        elif args.accelerate_method == "fora":
            model.timestep_map = {int(timestep): i for i, timestep in enumerate(diffusion.timestep_map)}
            model.set_reuse_policy("fora", args.fora_interval, args.the_first_half)

    if args.accelerate_method == "l2c":
        model.set_reuse_policy("l2c")
    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae_source, vae_local_only = resolve_vae_source(args.vae, args.vae_repo_path)
    if rank == 0:
        print(f"Loading VAE from {vae_source} (local_files_only={vae_local_only})")
    vae = AutoencoderKL.from_pretrained(vae_source, local_files_only=vae_local_only).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if args.accelerate_method == "l2c":
        router_name = os.path.basename(args.path).split('.')[0]
        folder_name = f"router-{router_name}-thres-{args.thres}-accelerate-{args.accelerate_method}-size-{args.image_size}-vae-{args.vae}-ddim-{args.ddim_sample}-" \
                      f"steps-{args.num_sampling_steps}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    elif args.accelerate_method == "fora":
        fora_suffix = "-firsthalf" if args.the_first_half else ""
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-psampler-{args.p_sample}-ddim-{args.ddim_sample}-" \
                  f"steps-{args.num_sampling_steps}-accelerate-fora-n-{args.fora_interval}{fora_suffix}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    else:
        folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-psampler-{args.p_sample}-ddim-{args.ddim_sample}-" \
                  f"steps-{args.num_sampling_steps}-accelerate-{args.accelerate_method}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if args.tcc_enable:
        tcc_tag = args.tcc_mode
        if args.tcc_lowrank:
            tcc_tag = f"{tcc_tag}_lowrank"
            if args.deprecated_lowrank:
                tcc_tag = f"{tcc_tag}_deprecated"
        if args.tcc_alpha != 0.5:
            tcc_tag = f"{tcc_tag}-alpha-{args.tcc_alpha}"
        if args.tcc_alpha_table:
            tcc_tag = f"{tcc_tag}-atable-{Path(args.tcc_alpha_table).stem}"
        if args.tcc_lowrank:
            tcc_tag = f"{tcc_tag}-eta-{args.tcc_eta}"
        if args.tcc_window != "12,18":
            tcc_tag = f"{tcc_tag}-window-{args.tcc_window.replace(',', '_')}"
        if args.tcc_lowrank:
            tcc_tag = f"{tcc_tag}-svd"
        sample_folder_dir = f"{sample_folder_dir}-tcc-{tcc_tag}"

    os.makedirs(f"{args.sample_dir}", exist_ok=True)
    if rank == 0 and args.save_to_disk:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        all_images = []

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    # -----------------------
    # TCC corrector (per-rank)
    # -----------------------
    tcc = None
    if args.tcc_enable:
        assert args.tcc_dir is not None, "TCC is enabled but --tcc-dir is not set"
        tcc_alpha = args.tcc_alpha
        if args.tcc_alpha_table is not None:
            tcc_alpha = torch.load(args.tcc_alpha_table, map_location="cpu")
            if isinstance(tcc_alpha, dict):
                tcc_alpha = tcc_alpha["alpha"]
        tcc = TccCorrector(
            tcc_dir=args.tcc_dir,
            device=torch.device(f"cuda:{device}"),
            cuda_id=device,
            preload=True,
            mode=args.tcc_mode,
            lowrank=args.tcc_lowrank,
            deprecated_lowrank=args.deprecated_lowrank,
            alpha=tcc_alpha,
            eta=args.tcc_eta,
            window=args.tcc_window,
            cache_only=True,
            apply_mode=args.tcc_apply_mode,
            num_steps=args.num_sampling_steps,
            subtract_prev_delta=not args.tcc_no_prev_delta_subtract,
        )
    dist.barrier()
    for _ in pbar:
        try:
            model.reset(args.num_sampling_steps)
        except TypeError:
            model.reset()
        
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale,tcc_corrector=tcc)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y,tcc_corrector=tcc)
            sample_fn = model.forward

        # Sample images:
        if args.p_sample:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        elif args.ddim_sample:
            ddim_kwargs = dict(
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
            if args.accelerate_method in ("l2c", "fora"):
                ddim_kwargs["is_sample"] = True
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, **ddim_kwargs
            )
        else:
            raise NotImplementedError
        
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to(dtype=torch.uint8)

        # Save samples to disk as individual .png files
        if args.save_to_disk:
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                sample = sample.cpu().numpy()
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        else:
            samples = samples.contiguous()
            gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples) 

            if rank == 0:
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        total += global_batch_size

        dist.barrier()

    # Make sure all processes have finished saving/collecting samples before rank 0
    # starts the potentially slow .npz build. Do not keep other ranks in a final
    # NCCL barrier while rank 0 reads/writes tens of GB of image data.
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        if args.save_to_disk:
            create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
            print("Done.")
        else:
            if rank == 0:
                arr = np.concatenate(all_images, axis=0)
                arr = arr[: args.num_fid_samples]

                out_path =  f"{sample_folder_dir}.npz"

                print(f"saving to {out_path}")
                np.savez(out_path, arr_0=arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-repo-path", type=str, default=None,
                        help="Optional local VAE directory or Hugging Face repo id. Overrides TCC_VAE_DIR and cache lookup.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", dest="tf32", action="store_true",
                        help="Use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--no-tf32", dest="tf32", action="store_false",
                        help="Disable TF32 matmuls.")
    parser.set_defaults(tf32=True)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    
    parser.add_argument("--ddim-sample", action="store_true", default=False,)
    parser.set_defaults(p_sample=False, l2c_enable=False, the_first_half=False)
    parser.add_argument("--accelerate-method", type=str, default=None, choices=["fora", "l2c"],
                        help="Use an accelerated model path. None means original DiT; l2c means Learn-to-Cache; fora means static FORA caching.")
    parser.add_argument("--fora-interval", type=float, default=2.0,
                        help="FORA cache interval N. Only used when --accelerate-method fora.")
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--path", type=str, default=None,)

    parser.add_argument("--save-to-disk", action="store_true", default=False,)
    parser.add_argument("--tcc-enable", dest="tcc_enable", action="store_true", help="Enable TCC corrector during sampling")
    parser.add_argument("--tcc-dir", dest="tcc_dir", metavar="TCC_DIR", type=str, default=None, help="Path to TCC pack directory containing step_XX.pt")
    parser.add_argument("--deprecated-lowrank", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tcc-alpha", dest="tcc_alpha", metavar="ALPHA", type=float, default=0.5)
    parser.add_argument("--tcc-window", dest="tcc_window", metavar="WINDOW", type=str, default="12,18",
                        help="step_min,step_max")
    parser.set_defaults(tcc_mode="tcc", tcc_lowrank=False, tcc_alpha_table=None, tcc_eta=1.0)
    parser.set_defaults(tcc_apply_mode="all", tcc_no_prev_delta_subtract=True)
    args = parser.parse_args()
    main(args)
