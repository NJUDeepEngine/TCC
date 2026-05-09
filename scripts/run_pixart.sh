#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXART_DIR="${ROOT}/tcc_pixart/pixart_alpha"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Default: single-GPU PixArt 256x20 FORA+TCC sampling, followed by writing
# sample images and a sample .npz.
#
# For the paper-style 4-GPU launch, set:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#
# Main command parameters:
#   --num-samples: number of prompts/images to generate before packing .npz.
#   --num-sampling-steps 20: paper default 256x20 setting.
#   --fresh-threshold 3: cache interval N. In PixArt, internal sampler steps
#     are indexed from 0 upward, opposite to the DiT-facing denoising-step
#     notation used elsewhere in the paper/code comments.
#   --cache-mode fora with --strict-force-fresh and --fresh-ratio 0: strict
#     FORA cache reuse line used by the main PixArt FORA+TCC setting.
#   --tcc-dir: directory produced by collect_pixart_tcc.sh.
#   --tcc-targets 1,2,4,5: strict FORA cache-step indices used by the 20-step
#     PixArt sampler when N=3. Steps 0,3,6 are full-fresh steps, so no TCC
#     pack is applied there. For another N, list only the actual cache steps in
#     the window you want to correct; the fresh steps are exactly the multiples
#     of N starting from 0.
#   --tcc-alpha 0.5: TCC correction strength for PixArt FORA+TCC.
#
# Native PixArt:
#   --native-pixart
#
# 50% steps baseline:
#   --native-pixart
#   --num-sampling-steps 10
#
# PixArt ToCa R=90:
#   --fresh-threshold 3
#   --fresh-ratio 0.10
#   --cache-type attention
#   --ratio-scheduler ToCa
#   --force-fresh global
#   --soft-fresh-weight 0.25
#   --no-tcc-stale-only
# Here tcc-stale-only controls whether TCC is applied only to stale reused
# tokens inside each ToCa cache step. The paper-default PixArt ToCa+TCC setting
# uses --no-tcc-stale-only, i.e. TCC is applied to the whole cached branch
# output at those ToCa cache steps.
# Here --ratio-scheduler is a built-in scheduler name, not a file path.
#
# PixArt ToCa R=60:
#   same ToCa block with --fresh-ratio 0.40
#
# PixArt ToCa+TCC:
#   use the ToCa block above, set --tcc-alpha 0.25, and use a ToCa TCC pack.

CMD=(
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port=29623
  "${PIXART_DIR}/scripts/sample_pixart.py"
  --distributed
  --sample-dir "${ROOT}/work/pixart_fora_tcc_samples"
  --txt-file "${COCO_PROMPTS:?set COCO_PROMPTS to the COCO-30K prompt text file}"
  --num-samples 30000
  --prompt-selection first
  --batch-size 100
  --seed 0
  --image-size 256
  --model-path "${PIXART_CKPT:?set PIXART_CKPT to PixArt-XL-2-256x256.pth}"
  --t5-path "${T5_DIR:?set T5_DIR to a directory containing t5-v1_1-xxl}"
  --tokenizer-path "${VAE_DIR:?set VAE_DIR to sd-vae-ft-ema}"
  --cfg-scale 4.5
  --num-sampling-steps 20
  --fresh-threshold 3
  --fresh-ratio 0
  --cache-type attention
  --cache-mode fora
  --strict-force-fresh
  --sample-ext jpg
  --tcc-dir "${TCC_PACK:?set TCC_PACK to a collected PixArt TCC pack}"
  --tcc-alpha 0.5
  --tcc-targets 1,2,4,5
  --tcc-stale-only
)

printf '[CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "${ROOT}/work/pixart_fora_tcc_samples"
"${CMD[@]}"
