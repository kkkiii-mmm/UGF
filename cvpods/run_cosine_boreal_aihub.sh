#!/usr/bin/env bash
#
# UGF ablation: MODEL.UGF.UNCERTAINTY_METHOD=cosine (see ugf.py), one run
# each for boreal and aihub with the same fixed seeds used for the
# raw_variance ablation, run back to back on a single GPU. aihub's
# RELIABILITY_ENABLED is forced False here (this directory's config.py
# default is True) so it matches boreal's False setting and the
# raw_variance run -- UNCERTAINTY_METHOD is the only thing that differs
# across the three (normalized_l2 default / raw_variance / cosine) runs.
# Everything else is left at each directory's own config.py defaults.
# Passed via cvpods' standard `--opts KEY VALUE` mechanism
# (tools/train_net.py); no cvpods core code or either experiment's
# config.py is touched.
#
# Usage:
#   ./run_cosine_boreal_aihub.sh
#   NUM_GPUS=1 CONDA_ENV=dense-teacher ./run_cosine_boreal_aihub.sh
#
set -uo pipefail

CVPODS_ROOT="/home/kimth/Harmonious-Teacher/cvpods"
CONDA_ENV="${CONDA_ENV:-dense-teacher}"
NUM_GPUS="${NUM_GPUS:-1}"

# Same TorchScript/sm_89 workaround used by this repo's other run scripts:
# dense-teacher's torch (1.10.1 / CUDA 11.3) predates the RTX 4080 SUPER
# (Ada, sm_89), so TorchScript's fused sigmoid_focal_loss_jit kernel crashes
# a few iterations in. PYTORCH_JIT=0 forces the eager fallback.
export PYTORCH_JIT=0

# conda's own activation scripts reference unset vars, which trips `set -u`.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

SUMMARY_LOG="$CVPODS_ROOT/cosine_boreal_aihub_summary.log"
: > "$SUMMARY_LOG"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_LOG"
}

# run_one <exp_dir> <tag> <seed> [extra --opts...]
run_one() {
    local exp_dir="$1" tag="$2" seed="$3"
    shift 3
    local extra_opts=("$@")
    local out_dir="$exp_dir/outputs/cosine/$tag"
    mkdir -p "$out_dir"

    log "===== [$tag] training -> $out_dir (seed=$seed, UNCERTAINTY_METHOD=cosine, extra_opts=${extra_opts[*]:-none}) ====="
    if ! (
        cd "$exp_dir" && \
        python3 "$CVPODS_ROOT/tools/train_net.py" \
            --dir . --num-gpus "$NUM_GPUS" \
            OUTPUT_DIR "$out_dir" \
            SEED "$seed" \
            MODEL.UGF.UNCERTAINTY_METHOD cosine \
            "${extra_opts[@]}"
    ) > "$out_dir/train.log" 2>&1; then
        log "===== [$tag] TRAINING FAILED (see $out_dir/train.log), skipping ====="
        return 1
    fi

    log "===== [$tag] DONE ====="
    return 0
}

failed_runs=()

run_one "$CVPODS_ROOT/ht_boreal_gating_linear" "boreal_cosine" 55408457 || failed_runs+=("boreal_cosine")
run_one "$CVPODS_ROOT/ht_aihub_gating_linear" "aihub_cosine" 13233898 MODEL.UGF.RELIABILITY_ENABLED False || failed_runs+=("aihub_cosine")

log "===== Both experiments attempted ====="
if [[ ${#failed_runs[@]} -gt 0 ]]; then
    log "Failed runs (${#failed_runs[@]}): ${failed_runs[*]}"
    exit 1
else
    log "All experiments completed successfully."
fi
