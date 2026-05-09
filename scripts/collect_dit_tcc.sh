#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Default: single-GPU DiT 256x20 FORA+TCC prior collection.
# For the paper-style 4-GPU launch, set:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#
# DIT_MODE only selects the implementation path and output folder name.
# Options: fora, l2c, toca.
DIT_MODE="${DIT_MODE:-fora}"

# Main command parameters:
#   --samples: number of prior latents collected per ImageNet class.
#   Step ids follow the reverse denoising order used by the sampler: a 20-step
#     run counts backward toward 0.
#   --target-steps 18,16,14,12: reverse sampling steps whose TCC packs are
#     collected for the DiT 256x20 FORA/L2C main setting.
#   --tcc-window 12,18: sample-step window where TCC is applied; all
#     transformer layers are used.
#   --fora-interval 2: FORA cache interval N.
#
# To collect L2C+TCC, set DIT_MODE=l2c and use this cache block in the command:
#   --accelerate-method l2c
#   --path "${ROUTER_CKPT:?set ROUTER_CKPT to an L2C router checkpoint}"
#   --thres 0.1
#
# To collect DiT-ToCa+TCC, set DIT_MODE=toca and use the ToCa collector
# arguments instead:
#   --output-dir "${ROOT}/work/dit_tcc_toca_pack"
#   --target-window 30,49
#   --samples-per-label 50
#   --batch-size 50
#   --fresh-threshold 2
#   --fresh-ratio 0.07
#   --ratio-scheduler ToCa-ddim50
#   --force-fresh global
#   --soft-fresh-weight 0.25
# Here --fresh-threshold is the cache interval N.  For DiT-ToCa main-table
# runs, TCC is applied later during 50-step sampling on window 49-30.
# Here --batch-size means the actual per-forward conditional batch size, while
# --samples-per-label is the total number of representatives collected for each
# ImageNet label.
#
# Optional ablation-only knobs, left at code defaults in the main command:
#   --tcc-pack-variant full|shift_only|scale_shift changes the stored
#     correction form.
#   --tcc-prior-pool-mode tokenpool|classpool|mixed changes how priors are
#     pooled before fitting TCC.
#   --tcc-prior-mix-classpool-ratio sets the classpool share for mixed pooling.
#   --tcc-prior-trajectory trajectory_consistent|one_shot changes whether
#     later packs are collected along the corrected trajectory.

case "${DIT_MODE}" in
  fora|l2c)
    COLLECTOR="${ROOT}/tcc_dit/cache_dit/pack_collection/compute_online_tcc.py"
    ;;
  toca)
    COLLECTOR="${ROOT}/tcc_dit/toca_dit/pack_collection/compute_online_tcc.py"
    ;;
  *)
    echo "DIT_MODE must be one of: fora, l2c, toca" >&2
    exit 2
    ;;
esac

CMD=(
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port=29531
  "${COLLECTOR}"
  --out-dir "${ROOT}/work/dit_tcc_${DIT_MODE}_pack"
  --model DiT-XL/2
  --image-size 256
  --num-classes 1000
  --samples 100
  --batch-size 50
  --cfg-scale 1.5
  --num-sampling-steps 20
  --target-steps 18,16,14,12
  --ckpt "${CKPT:?set CKPT to a DiT-XL/2 checkpoint}"
  --accelerate-method fora
  --fora-interval 2
  --tcc-alpha 1.75
  --tcc-window 12,18
  --global-seed 0
  --device cuda:0
  --init-states
  --tf32
)

printf '[CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
