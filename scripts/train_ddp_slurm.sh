#!/bin/bash
#SBATCH --job-name=langsplat-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=langsplat_ddp_%j.out
#SBATCH --error=langsplat_ddp_%j.err

#
# LangSplat DDP Training Script - SLURM Cluster
#
# Submit with: sbatch scripts/train_ddp_slurm.sh
#
# Modify the SBATCH directives above according to your cluster configuration.
#

# ============================================
# Configuration - MODIFY THESE
# ============================================
SOURCE_PATH="/path/to/your/dataset"
MODEL_PATH="/path/to/output/model"
CHECKPOINT_PATH="/path/to/rgb_checkpoint.pth"
FEATURE_LEVEL=0  # 0, 1, or 2

# ============================================
# SLURM Environment Setup
# ============================================

# Get master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Calculate world size
GPUS_PER_NODE=8
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

echo "=============================================="
echo "LangSplat DDP Training - SLURM Cluster"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node list:       $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Master address:  $MASTER_ADDR"
echo "Master port:     $MASTER_PORT"
echo "GPUs per node:   $GPUS_PER_NODE"
echo "World size:      $WORLD_SIZE"
echo "=============================================="
echo "Source path:     $SOURCE_PATH"
echo "Model path:      $MODEL_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Feature level:   $FEATURE_LEVEL"
echo "=============================================="

# Set environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Activate your conda environment if needed
# source activate langsplat

# Change to the LangSplat directory
cd $SLURM_SUBMIT_DIR

# Launch training using srun + torchrun
srun --exclusive \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
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

echo "Training complete!"
