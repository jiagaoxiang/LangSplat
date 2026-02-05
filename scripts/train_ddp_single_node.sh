#!/bin/bash
#
# LangSplat DDP Training Script - Single Node (8 GPUs)
#
# Usage:
#   ./scripts/train_ddp_single_node.sh <source_path> <model_path> <checkpoint_path> <feature_level> [num_gpus]
#
# Example:
#   ./scripts/train_ddp_single_node.sh ./data/figurines ./output/figurines ./output/figurines/chkpnt30000.pth 0 8
#
# Arguments:
#   source_path     - Path to the dataset (COLMAP or Blender format)
#   model_path      - Path to save the trained model
#   checkpoint_path - Path to the RGB-trained checkpoint (required for language feature training)
#   feature_level   - Feature level to train (0, 1, or 2)
#   num_gpus        - Number of GPUs to use (default: 8)
#

set -e

# Parse arguments
SOURCE_PATH=${1:?Error: source_path is required}
MODEL_PATH=${2:?Error: model_path is required}
CHECKPOINT_PATH=${3:?Error: checkpoint_path is required}
FEATURE_LEVEL=${4:?Error: feature_level is required}
NUM_GPUS=${5:-8}

# Validate feature level
if [[ ! "$FEATURE_LEVEL" =~ ^[0-2]$ ]]; then
    echo "Error: feature_level must be 0, 1, or 2"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "=============================================="
echo "LangSplat DDP Training - Single Node"
echo "=============================================="
echo "Source path:     $SOURCE_PATH"
echo "Model path:      $MODEL_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Feature level:   $FEATURE_LEVEL"
echo "Number of GPUs:  $NUM_GPUS"
echo "=============================================="

# Set master address and port for single node
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch training with torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_ddp.py \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --start_checkpoint "$CHECKPOINT_PATH" \
    --feature_level $FEATURE_LEVEL \
    --iterations 30000 \
    --save_iterations 7000 30000 \
    --checkpoint_iterations 7000 30000 \
    --test_iterations 7000 30000

echo "Training complete!"
