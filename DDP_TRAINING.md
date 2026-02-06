# LangSplat DDP (Distributed Data Parallel) Training

This document describes how to use DDP for distributed training of LangSplat's language feature training across multiple GPUs.

## Overview

DDP training is supported **only for language feature training** (`--include_feature`), not for RGB/geometry training. This is because:

1. **Language feature training** has a fixed model structure (no densification)
2. Only the `_language_feature` parameter is trained (geometry is frozen)
3. This makes DDP straightforward - gradients are simply averaged across GPUs

For RGB/geometry training (with densification), use the original `train.py` with a single GPU.

## Training Workflow

The complete LangSplat training workflow is:

1. **Stage 1: RGB Training (Single GPU)**
   ```bash
   python train.py -s <source_path> -m <model_path> --no_include_feature
   ```
   This trains the 3D Gaussian geometry with densification enabled.

2. **Stage 2: Language Feature Training (Multi-GPU DDP)**
   ```bash
   # Single node with 8 GPUs
   ./scripts/train_ddp_single_node.sh <source_path> <model_path> <rgb_checkpoint> <feature_level> 8
   ```
   This trains the language features using the frozen geometry from Stage 1.

## Requirements

- PyTorch with distributed support
- NCCL backend (default, recommended for NVIDIA GPUs)
- All GPUs should have access to the dataset and checkpoint files

## Single Node Training (8 GPUs)

### Using the Launch Script

```bash
./scripts/train_ddp_single_node.sh \
    ./data/figurines \
    ./output/figurines \
    ./output/figurines/chkpnt30000.pth \
    0 \    # feature_level (0, 1, or 2)
    8      # number of GPUs
```

### Using torchrun Directly

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_ddp.py \
    -s ./data/figurines \
    -m ./output/figurines \
    --start_checkpoint ./output/figurines/chkpnt30000.pth \
    --feature_level 0 \
    --iterations 30000
```

## Multi-Node Training

### Example: 2 Nodes with 8 GPUs Each (16 GPUs Total)

**On Node 0 (Master):**
```bash
./scripts/train_ddp_multi_node.sh \
    ./data/figurines \
    ./output/figurines \
    ./output/figurines/chkpnt30000.pth \
    0 \              # feature_level
    2 \              # num_nodes
    0 \              # node_rank (this is node 0)
    192.168.1.100 \  # master_addr (IP of this node)
    8 \              # gpus_per_node
    29500            # master_port
```

**On Node 1:**
```bash
./scripts/train_ddp_multi_node.sh \
    ./data/figurines \
    ./output/figurines \
    ./output/figurines/chkpnt30000.pth \
    0 \              # feature_level
    2 \              # num_nodes
    1 \              # node_rank (this is node 1)
    192.168.1.100 \  # master_addr (IP of node 0)
    8 \              # gpus_per_node
    29500            # master_port
```

### SLURM Cluster

For SLURM-managed clusters, modify and submit `scripts/train_ddp_slurm.sh`:

```bash
# Edit the configuration in the script first
sbatch scripts/train_ddp_slurm.sh
```

## Training All Feature Levels

LangSplat requires training 3 feature levels (0, 1, 2). You can train them in parallel on different GPU groups or sequentially:

```bash
# Sequential training of all levels
for level in 0 1 2; do
    ./scripts/train_ddp_single_node.sh \
        ./data/figurines \
        ./output/figurines \
        ./output/figurines/chkpnt30000.pth \
        $level \
        8
done
```

## How DDP Works

### Camera Distribution

Each GPU processes different cameras in parallel:
- GPU 0: cameras [0, 8, 16, ...]
- GPU 1: cameras [1, 9, 17, ...]
- GPU 2: cameras [2, 10, 18, ...]
- ...

This covers all cameras when the count is divisible by the number of GPUs.
If not, the last uneven cameras are dropped each epoch to keep per-rank work equal.

### Gradient Synchronization

After each forward-backward pass:
1. Each GPU computes gradients for its camera
2. Gradients are averaged across all GPUs using `all_reduce`
3. All GPUs perform the same optimizer step

This results in effectively `N` times larger batch size (where `N` is the number of GPUs).

### Logging and Checkpointing

- Only rank 0 (master) handles:
  - TensorBoard logging
  - Progress bar display
  - Model checkpointing
  - Saving final model

## Troubleshooting

### NCCL Timeout

If you see NCCL timeout errors, try:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Use your network interface
```

### Out of Memory

Each GPU needs memory for:
- Full model parameters (shared across GPUs)
- Gradients for one camera view
- Optimizer states

If OOM occurs, try reducing image resolution in the dataset.

### Slow Training

- Ensure GPUs are on the same network for multi-node
- Use NCCL backend (default) for best performance
- Check that all GPUs have similar utilization

## Performance Expectations

With 8 GPUs on a single node:
- ~8x throughput for camera processing
- Same per-iteration time (bounded by gradient sync)
- Effective batch size is 8x larger
- May converge faster with adjusted learning rate

## Files Created

- `train_ddp.py` - Main DDP training script
- `utils/distributed_utils.py` - DDP utility functions
- `scripts/train_ddp_single_node.sh` - Single node launch script
- `scripts/train_ddp_multi_node.sh` - Multi-node launch script
- `scripts/train_ddp_slurm.sh` - SLURM cluster job script
