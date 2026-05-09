# TCC Reproducibility Code

This repository contains the release code paths for the DiT and PixArt TCC
experiments.  The package focuses on generating TCC packs, sampling images, and
writing sample `.npz` files.  FID, sFID, IS, precision/recall, and CLIP Score
can be computed with standard external evaluation tools from the generated
`.npz` files.

## Environment

The experiments use ordinary DiT/PixArt-style Python environments.  The files
below list only the packages that matter for this release; install CUDA-enabled
PyTorch in the way recommended for your system.

```bash
conda create -n tcc_dit python=3.10
conda activate tcc_dit
pip install -r envs/requirements_dit.txt

conda create -n tcc_pixart python=3.10
conda activate tcc_pixart
pip install -r envs/requirements_pixart.txt
```

The DiT environment is used for DiT FORA/L2C and DiT-ToCa.  PixArt is kept in a
separate environment because it depends on T5/transformers and xFormers.

## Launch Convention

All release scripts default to a single-GPU launch:

```bash
CUDA_VISIBLE_DEVICES=0
NPROC_PER_NODE=1
```

This keeps the out-of-the-box behavior simple for first-time users.  For the
paper-style 4-GPU runs, set these explicitly before calling a script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC_PER_NODE=4
```

## Code Layout

- `tcc_dit/cache_dit/`: DiT cache branch for native DiT, FORA, L2C, FORA+TCC,
  and L2C+TCC.
- `tcc_dit/toca_dit/`: DiT-ToCa branch for ToCa and ToCa+TCC.
- `tcc_pixart/pixart_alpha/`: PixArt branch for native PixArt, FORA, ToCa, and
  the corresponding TCC variants.
- `scripts/`: minimal command templates.  The defaults are 256x20 FORA+TCC;
  edit the command flags directly for other paper settings.

## External Assets

Large assets are not included.

For DiT:

- DiT-XL/2 checkpoints, e.g. `DiT-XL-2-256x256.pt` and
  `DiT-XL-2-512x512.pt`.
- Stable Diffusion VAE, e.g. `stabilityai/sd-vae-ft-ema` or a local equivalent.
- L2C router checkpoints for L2C and L2C+TCC runs.

For PixArt:

- PixArt-alpha checkpoint, e.g. `PixArt-XL-2-256x256.pth`.
- T5 text encoder directory containing `t5-v1_1-xxl`.
- Stable Diffusion VAE directory.
- MS-COCO prompt file for 30K sampling.

Generated samples, generated `.npz` files, and generated TCC packs are not
included.  PixArt representative prompt files for TCC collection are included in
`tcc_pixart/txt/`.

## DiT

Collect a DiT TCC pack:

```bash
CKPT=/path/to/DiT-XL-2-256x256.pt bash scripts/collect_dit_tcc.sh
```

The script is a plain template.  It defaults to DiT 256x20 FORA+TCC.  The top
comments explain the important flags and the minimal edits for L2C and
DiT-ToCa.  L2C needs `ROUTER_CKPT=/path/to/router.pt`.

For the DiT 256x20 FORA/L2C main-table setting, the collector stores packs for
reverse steps `18,16,14,12`, while TCC is applied during sampling on window
`19-12`.

The main DiT ablation controls are exposed by the collector and summarized in
`scripts/collect_dit_tcc.sh`:

- `--tcc-pack-variant full|shift_only|scale_shift`
- `--tcc-prior-pool-mode tokenpool|classpool|mixed`
- `--tcc-prior-mix-classpool-ratio`
- `--tcc-prior-trajectory trajectory_consistent|one_shot`

Leaving these unchanged gives the main TCC behavior.

Sample DiT outputs and write a sample `.npz`:

```bash
CKPT=/path/to/DiT-XL-2-256x256.pt \
TCC_PACK=/path/to/tcc_pack \
bash scripts/run_dit.sh
```

The script defaults to DiT 256x20 FORA+TCC.  By default it writes the packed
`.npz` needed for evaluation and keeps image tensors in memory instead of
writing 50K individual `.png` files.  If you also want per-sample `.png` files,
launch with `SAVE_TO_DISK=1`.  For no-cache DiT, cache baselines, L2C, or
DiT-ToCa, edit the flag blocks described at the top of the script.

`--fresh-threshold` is ToCa's cache interval N, and `--fresh-ratio` controls the
fresh-token ratio.  The DiT-ToCa main-table setting uses 50 sampling steps,
`--fresh-threshold 2`, `--fresh-ratio 0.07`, and applies TCC on the 49-30
window.  In the DiT-ToCa collector, `--batch-size` means the actual per-forward
conditional batch size, while `--samples-per-label` is the total number of
representatives collected for each ImageNet label.

## PixArt

Collect a PixArt TCC pack:

```bash
PIXART_CKPT=/path/to/PixArt-XL-2-256x256.pth \
T5_DIR=/path/to/t5-cache-root \
VAE_DIR=/path/to/sd-vae-ft-ema \
bash scripts/collect_pixart_tcc.sh
```

The script defaults to PixArt 256x20 FORA+TCC.  The top comments explain the
important flags and the small edits needed for PixArt ToCa+TCC collection.

For the prompt-count ablation, edit `--txt-file` and `--num-prompts` in the
collector to use:

- `tcc_pixart/txt/coco2017_train_prior_1000.txt`
- `tcc_pixart/txt/coco2017_train_prior_2000.txt`
- `tcc_pixart/txt/coco2017_train_prior_5000.txt`

Sample PixArt images and write a sample `.npz`:

```bash
PIXART_CKPT=/path/to/PixArt-XL-2-256x256.pth \
T5_DIR=/path/to/t5-cache-root \
VAE_DIR=/path/to/sd-vae-ft-ema \
COCO_PROMPTS=/path/to/COCO_caption_prompts_first30000.txt \
TCC_PACK=/path/to/pixart_tcc_pack \
bash scripts/run_pixart.sh
```

Important PixArt switches are documented at the top of `scripts/run_pixart.sh`.
In short, `--native-pixart` bypasses caching, `--num-sampling-steps 10` gives the
50% steps row, `--fresh-ratio 0.10` corresponds to ToCa R=90, and
`--fresh-ratio 0.40` corresponds to ToCa R=60.
