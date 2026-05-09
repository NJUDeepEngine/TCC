#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXART_DIR="${ROOT}/tcc_pixart/pixart_alpha"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Default: single-GPU PixArt 256x20 FORA+TCC prior collection.
# For the paper-style 4-GPU launch, set:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NPROC_PER_NODE=4
#
# PIXART_MODE only selects the collector path and output folder name.
# Options: fora, toca.
PIXART_MODE="${PIXART_MODE:-fora}"

# Main command parameters:
#   --targets 1,2,4,5: strict FORA cache-step indices for the 20-step PixArt
#     sampler when N=3. They are implementation step ids, not layer ids.
#     PixArt counts internal sampler steps from 0 upward, opposite to the
#     DiT-facing denoising-step notation used elsewhere in the repo. Steps
#     0,3,6 are full-fresh steps and therefore do not produce FORA packs. For
#     another N, list only the cache steps you want to collect; fresh steps are
#     exactly the multiples of N starting from 0.
#   --num-prompts: number of prompts used to estimate the TCC prior.
#   --fresh-threshold 3: cache interval N.
#   --tcc-alpha 0.5: TCC correction strength for PixArt FORA+TCC.
#
# Prompt-count ablation:
#   edit --txt-file and --num-prompts to use the 1k, 2k, or 5k prompt files
#   under tcc_pixart/txt/.
#
# To collect PixArt ToCa+TCC, set PIXART_MODE=toca and use this ToCa block in
# place of the FORA target/cache lines:
#   --target-window 0,10
#   --fresh-threshold 3
#   --fresh-ratio 0.10
#   --cache-type attention
#   --ratio-scheduler ToCa
#   --force-fresh global
#   --soft-fresh-weight 0.25
#   --tcc-alpha 0.25
#   --no-stale-only
# Here stale-only controls whether TCC is estimated/applied only on stale
# reused tokens within a ToCa cache step. The paper-default PixArt ToCa+TCC
# setting uses --no-stale-only, i.e. TCC is applied to the whole cached branch
# output at those ToCa cache steps rather than only the stale-token subset.
# Here --target-window is the contiguous sampler-step window used for ToCa TCC
# pack collection; it uses the same reverse step numbering.  --ratio-scheduler
# is a built-in scheduler name, not a file path.  --fresh-ratio 0.10
# corresponds to ToCa R=90.

case "${PIXART_MODE}" in
  fora)
    COLLECTOR="${PIXART_DIR}/scripts/collect_fora_tcc_pack.py"
    ;;
  toca)
    COLLECTOR="${PIXART_DIR}/scripts/collect_online_tcc.py"
    ;;
  *)
    echo "PIXART_MODE must be one of: fora, toca" >&2
    exit 2
    ;;
esac

CMD=(
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  python -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port=29621
  "${COLLECTOR}"
  --output-dir "${ROOT}/work/pixart_${PIXART_MODE}_tcc_pack"
  --txt-file "${ROOT}/tcc_pixart/txt/coco2017_train_prior_1000.txt"
  --num-prompts 1000
  --prompt-selection first
  --targets 1,2,4,5
  --batch-size 8
  --seed 0
  --image-size 256
  --model-path "${PIXART_CKPT:?set PIXART_CKPT to PixArt-XL-2-256x256.pth}"
  --t5-path "${T5_DIR:?set T5_DIR to a directory containing t5-v1_1-xxl}"
  --tokenizer-path "${VAE_DIR:?set VAE_DIR to sd-vae-ft-ema}"
  --cfg-scale 4.5
  --num-sampling-steps 20
  --fresh-threshold 3
  --cache-type attention
  --tcc-alpha 0.5
  --stale-only
)

printf '[CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
