#!/bin/bash
#
# LangSplat DDP Training Script - Multi-Node
#
# This script should be run on EACH node in the cluster.
# Make sure all nodes can communicate with each other.
#
# Usage:
#   ./scripts/train_ddp_multi_node.sh <source_path> <model_path> <checkpoint_path> <feature_level> \
#       <num_nodes> <node_rank> <master_addr> [gpus_per_node] [master_port]
#
# Example (2 nodes, 8 GPUs each):
#   # On node 0 (master):
#   ./scripts/train_ddp_multi_node.sh ./data/figurines ./output/figurines ./output/figurines/chkpnt30000.pth 0 2 0 node0-ip 8
#
#   # On node 1:
#   ./scripts/train_ddp_multi_node.sh ./data/figurines ./output/figurines ./output/figurines/chkpnt30000.pth 0 2 1 node0-ip 8
#
# Arguments:
#   source_path     - Path to the dataset (COLMAP or Blender format)
#   model_path      - Path to save the trained model
#   checkpoint_path - Path to the RGB-trained checkpoint
#   feature_level   - Feature level to train (0, 1, or 2)
#   num_nodes       - Total number of nodes
#   node_rank       - Rank of this node (0-indexed)
#   master_addr     - IP address of the master node (rank 0)
#   gpus_per_node   - Number of GPUs per node (default: 8)
#   master_port     - Port for distributed communication (default: 29500)
#

set -e

# Parse arguments
SOURCE_PATH=${1:?Error: source_path is required}
MODEL_PATH=${2:?Error: model_path is required}
CHECKPOINT_PATH=${3:?Error: checkpoint_path is required}
FEATURE_LEVEL=${4:?Error: feature_level is required}
NUM_NODES=${5:?Error: num_nodes is required}
NODE_RANK=${6:?Error: node_rank is required}
MASTER_ADDR=${7:?Error: master_addr is required}
GPUS_PER_NODE=${8:-8}
MASTER_PORT=${9:-29500}

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

WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "=============================================="
echo "LangSplat DDP Training - Multi-Node"
echo "=============================================="
echo "Source path:     $SOURCE_PATH"
echo "Model path:      $MODEL_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Feature level:   $FEATURE_LEVEL"
echo "Number of nodes: $NUM_NODES"
echo "This node rank:  $NODE_RANK"
echo "Master address:  $MASTER_ADDR"
echo "Master port:     $MASTER_PORT"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "Total world size: $WORLD_SIZE"
echo "=============================================="

# Set environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Launch training with torchrun
torchrun \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_ddp.py \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --start_checkpoint "$CHECKPOINT_PATH" \
    --feature_level $FEATURE_LEVEL \
    --iterations 30000 \
    --save_iterations 7000 30000 \
    --checkpoint_iterations 7000 30000 \
    --test_iterations 7000 30000

echo "Training complete on node $NODE_RANK!"
