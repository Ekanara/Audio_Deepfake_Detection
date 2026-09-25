#!/usr/bin/env bash
# train.sh — 2-stage BEATs training for LL_fake_only (standalone)
#
# Works from anywhere — paths resolve relative to this script's folder.
#   bash train.sh [1|2|both]      (default: both)
#
# Examples:
#   bash LL_fake_only/train.sh both
#   WANDB_API=your_key bash LL_fake_only/train.sh both

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# train.py defaults (data dir, output dir) already point inside this folder.
run() {
    python "$HERE/train.py" --stage "$1" ${WANDB_API:+--wandb_api "$WANDB_API"}
}

case "${1:-both}" in
    1|stage1) log "Stage 1";        run 1 ;;
    2|stage2) log "Stage 2";        run 2 ;;
    both)     log "Stage 1 + 2";    run both ;;
    *)
        echo "Usage: bash train.sh [1|2|both]"
        exit 1 ;;
esac

log "Done! Checkpoints in $HERE/checkpoint/"
log "Re-run precompute_reference.py so detect.py uses the new model:"
log "  python $HERE/precompute_reference.py"
