#!/usr/bin/env bash
# infer.sh — full evaluation on the bundled test sets (standalone)
#
# Works from anywhere — paths resolve relative to this script's folder.
#   bash infer.sh
#
# Uses a freshly trained stage 2 checkpoint if present, else the bundled one.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Prefer a trained checkpoint over the bundled one.
CKPT=$(ls -t "$HERE"/checkpoint/ll_fake_only_stage2/sample-*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ]; then
    CKPT="$HERE/checkpoint/ll_fake_only_stage2.ckpt"
    log "Using bundled checkpoint: $CKPT"
else
    log "Using trained checkpoint: $CKPT"
fi

python "$HERE/infer.py" \
    --checkpoint "$CKPT" \
    --fake_ref   "$HERE/data/Event_train_stage1_fakeonly.json" \
    --test_json  "$HERE/data/test_track2.json" \
                 "$HERE/data/Event_test_5class.json" \
                 "$HERE/data/TUTASC19_test_5class.json" \
    --output_dir "$HERE/inference_outputs" \
    --plot --save_csv

log "Outputs saved to $HERE/inference_outputs/"
