#!/bin/bash
# stage1 profile_all: peak -> E2/E3/E5 -> E4 -> E1 (10fps, 30fps) -> peak
# Usage: bash scripts/profiling/run_all.sh <out_dir> [--smoke]
set -euo pipefail

OUT_DIR=$(realpath -m "$1")
SMOKE=${2:-}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mediapipe
mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
    --format=csv -l 1 > "$OUT_DIR/gpu_monitor.csv" &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

python peak.py "$OUT_DIR" --unit peak_before $SMOKE
python bench_model.py "$OUT_DIR" $SMOKE
python profile_kernels.py "$OUT_DIR" $SMOKE
python latency_breakdown.py "$OUT_DIR" --fps 10 $SMOKE
python latency_breakdown.py "$OUT_DIR" --fps 30 $SMOKE
python peak.py "$OUT_DIR" --unit peak_after $SMOKE
