#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SAVE_TO_DISK="${SAVE_TO_DISK:-0}"

# Default: single-GPU DiT 256x20 FORA+TCC sampling, followed by writing a
# sample .npz. By default, individual .png files are not written, which keeps
# the evaluation path faster and lighter on disk. Set SAVE_TO_DISK=1 if you
# also want per-sample .png files for manual inspection.
#
# For the paper-style 4-GPU launch, set:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#
# DIT_MODE only selects the implementation path and output folder name.
# Options: baseline, fora, l2c, toca.
DIT_MODE="${DIT_MODE:-fora}"

# Main command parameters:
#   --num-fid-samples: number of images to generate before packing .npz.
#   --num-sampling-steps 20: paper default 256x20 setting.
#   Step ids follow the reverse denoising order used by the sampler: a 20-step
#     run counts backward toward 0.
#   --tcc-dir: directory produced by collect_dit_tcc.sh.
#   --tcc-window 12,18: sample-step window where TCC is applied; all
#     transformer layers are used.
#   --tcc-alpha: TCC correction strength.
#
# Native DiT uses the same command without the FORA block and without the TCC
# block:
#   --accelerate-method fora
#   --fora-interval 2
#   --tcc-enable
#   --tcc-dir "${TCC_PACK:?set TCC_PACK to a collected DiT TCC pack}"
#   --tcc-alpha 1.75
#   --tcc-window 12,18
#
# FORA without TCC keeps:
#   --accelerate-method fora
#   --fora-interval 2
#
# L2C+TCC: set DIT_MODE=l2c and use this cache block:
#   --accelerate-method l2c
#   --path "${ROUTER_CKPT:?set ROUTER_CKPT to an L2C router checkpoint}"
#   --thres 0.1
#
# DiT-ToCa+TCC: set DIT_MODE=toca and use a ToCa cache block instead of the
# FORA block:
#   --num-sampling-steps 50
#   --fresh-threshold 2
#   --fresh-ratio 0.07
#   --cache-type attention
#   --ratio-scheduler ToCa-ddim50
#   --force-fresh global
#   --soft-fresh-weight 0.25
#   --tcc-alpha 0.5
#   --tcc-targets 31
# Here --fresh-threshold is the cache interval N.  The DiT-ToCa main-table
# setting uses 50-step sampling and applies TCC in the 49-30 window.

case "${DIT_MODE}" in
  baseline|fora|l2c)
    SAMPLER="${ROOT}/tcc_dit/cache_dit/sample_ddp.py"
    ;;
  toca)
    SAMPLER="${ROOT}/tcc_dit/toca_dit/sample_ddp.py"
    ;;
  *)
    echo "DIT_MODE must be one of: baseline, fora, l2c, toca" >&2
    exit 2
    ;;
esac

CMD=(
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port=29541
  "${SAMPLER}"
  --model DiT-XL/2
  --vae ema
  --sample-dir "${ROOT}/work/dit_${DIT_MODE}_samples"
  --per-proc-batch-size 25
  --num-fid-samples 50000
  --image-size 256
  --num-classes 1000
  --cfg-scale 1.5
  --num-sampling-steps 20
  --global-seed 0
  --ckpt "${CKPT:?set CKPT to a DiT-XL/2 checkpoint}"
  --ddim-sample
  --accelerate-method fora
  --fora-interval 2
  --tcc-enable
  --tcc-dir "${TCC_PACK:?set TCC_PACK to a collected DiT TCC pack}"
  --tcc-alpha 1.75
  --tcc-window 12,18
  --tf32
)

if [[ "${SAVE_TO_DISK}" == "1" ]]; then
  CMD+=(--save-to-disk)
fi

printf '[CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "${ROOT}/work"
"${CMD[@]}"
