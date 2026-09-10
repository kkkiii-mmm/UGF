#!/usr/bin/env bash
# Resume aihub_cosine after the machine's unexpected reboot killed the
# original run at iter 35350/40000. --resume picks up model_iter_0030000.pth
# via OUTPUT_DIR's last_checkpoint file (the only periodic checkpoint saved
# past that point). Same seed/opts as the original invocation.
set -uo pipefail

CVPODS_ROOT="/home/kimth/Harmonious-Teacher/cvpods"
CONDA_ENV="${CONDA_ENV:-dense-teacher}"
NUM_GPUS="${NUM_GPUS:-1}"
EXP_DIR="$CVPODS_ROOT/ht_aihub_gating_linear"
OUT_DIR="$EXP_DIR/outputs/cosine/aihub_cosine"

export PYTORCH_JIT=0
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

SUMMARY_LOG="$CVPODS_ROOT/cosine_boreal_aihub_summary.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"; }

log "===== [aihub_cosine] RESUMING after reboot (from iter 30000) -> $OUT_DIR ====="
if ! (
    cd "$EXP_DIR" && \
    python3 "$CVPODS_ROOT/tools/train_net.py" \
        --dir . --num-gpus "$NUM_GPUS" --resume \
        OUTPUT_DIR "$OUT_DIR" \
        SEED 13233898 \
        MODEL.UGF.UNCERTAINTY_METHOD cosine \
        MODEL.UGF.RELIABILITY_ENABLED False
) >> "$OUT_DIR/train.log" 2>&1; then
    log "===== [aihub_cosine] RESUMED TRAINING FAILED (see $OUT_DIR/train.log) ====="
    exit 1
fi

log "===== [aihub_cosine] DONE (resumed) ====="
