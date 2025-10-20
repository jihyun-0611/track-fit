#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASE_SOURCE[0]}")" && pwd)"

CONFIG="${PROJECT_ROOT}/configs/exercise/j.py"

GPUS=1
PORT=$((12000 + RANDOM % 20000))


VALIDATE="--validate"
TEST_BEST="--test-best"
TEST_LAST="--test-last"

echo "======================================================"
echo "ProtoGCN Fine-tuning"
echo "======================================================"

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found!: $CONFIG"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/data/exercise_dataset.pkl" ]; then
    echo "Dataset not found!: ${PROJECT_ROOT}/data/exercise_dataset.pkl"
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/checkpoints/finegym_j/best_top1_acc_epoch_141.pth" ]; then
    echo "Pretrained model not found!: ${PROJECT_ROOT}/checkpoints/finegym_j/best_top1_acc_epoch_141.pth"
    echo "Training from scratch..."
fi

echo ""
echo "Config: $CONFIG"
echo "GPUS: $GPUS"
echo "PORT: $PORT"
echo ""

PROTOGCN_DIR="${PROJECT_ROOT}/external/ProtoGCN"
export PYTHONPATH="${PROJECT_ROOT}:${PROTOGCN_DIR}:${PYTHONPATH}" 

export MKL_SERVICE_FORCE_INTEL=1
cd "${PROTOGCN_DIR}" || exit 1

echo "Starting training..."
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    "$CONFIG" \
    --launcher pytorch \
    $VALIDATE \
    $TEST_BEST \
    $TEST_LAST \

cd "${PROJECT_ROOT}" || exit 1

echo "======================================================"
echo "Training done."
echo "Results saved to: work_dirs/exercise/j/"
echo "======================================================"