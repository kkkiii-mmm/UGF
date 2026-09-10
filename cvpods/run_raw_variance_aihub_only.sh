#!/usr/bin/env bash
# Standalone aihub-only relaunch for the raw_variance ablation, with
# RELIABILITY_ENABLED forced False (see run_raw_variance_boreal_aihub.sh).
# Used to correct the aihub run without touching boreal's already-running
# process (bash caches run_one's body in memory, so editing the combined
# script mid-run would not affect its already-launched aihub invocation).
set -uo pipefail

CVPODS_ROOT="/home/kimth/Harmonious-Teacher/cvpods"
CONDA_ENV="${CONDA_ENV:-dense-teacher}"
NUM_GPUS="${NUM_GPUS:-1}"
EXP_DIR="$CVPODS_ROOT/ht_aihub_gating_linear"
OUT_DIR="$EXP_DIR/outputs/raw_variance/aihub_raw_variance"
SEED=13233898

export PYTORCH_JIT=0
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

SUMMARY_LOG="$CVPODS_ROOT/raw_variance_boreal_aihub_summary.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"; }

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

log "===== [aihub_raw_variance] (corrected relaunch, RELIABILITY_ENABLED=False) training -> $OUT_DIR (seed=$SEED, UNCERTAINTY_METHOD=raw_variance) ====="
if ! (
    cd "$EXP_DIR" && \
    python3 "$CVPODS_ROOT/tools/train_net.py" \
        --dir . --num-gpus "$NUM_GPUS" \
        OUTPUT_DIR "$OUT_DIR" \
        SEED "$SEED" \
        MODEL.UGF.UNCERTAINTY_METHOD raw_variance \
        MODEL.UGF.RELIABILITY_ENABLED False
) > "$OUT_DIR/train.log" 2>&1; then
    log "===== [aihub_raw_variance] TRAINING FAILED (see $OUT_DIR/train.log) ====="
    exit 1
fi

log "===== [aihub_raw_variance] DONE ====="
