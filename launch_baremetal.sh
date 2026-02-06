#!/bin/bash
#
# LangSplat Bare Metal Multi-Node DDP Training Launcher
# ======================================================
# This script launches distributed DDP training for LangSplat
# language feature training across multiple bare metal nodes
# without requiring SLURM, Kubernetes, or other job schedulers.
# It handles Docker container setup on each node automatically.
#
# Usage:
#   Option 1: Using nodes file with --num-nodes (recommended)
#     ./launch_baremetal.sh --num-nodes 2 [--nodes-file /path/to/nodes.txt]
#
#   Option 2: Explicit node list
#     ./launch_baremetal.sh --nodes "node1,node2,node3" [--gpus-per-node 8]
#
#   Option 3: Manual launch on each node (no SSH, inside container)
#     On master (node 0):
#       MASTER_ADDR=<master_ip> NNODES=2 NODE_RANK=0 ./launch_baremetal.sh --inside-container
#     On worker (node 1):
#       MASTER_ADDR=<master_ip> NNODES=2 NODE_RANK=1 ./launch_baremetal.sh --inside-container
#
# Environment Variables (for manual mode):
#   MASTER_ADDR     - IP address of the master node (required for workers)
#   MASTER_PORT     - Port for distributed communication (default: 29500)
#   NNODES          - Total number of nodes (default: 1)
#   NODE_RANK       - Rank of current node, 0-indexed (default: 0)
#   GPUS_PER_NODE   - Number of GPUs per node (default: 8)
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Default values
# ============================================================
DEFAULT_MASTER_PORT=29500
DEFAULT_GPUS_PER_NODE=8
DEFAULT_DOCKER_IMAGE="rocm/primus:v26.1"
# Default nodes file: shared MI325 cluster nodes
USER_HOME="/home/douglas.jia@amd.com"
DEFAULT_NODES_FILE="${USER_HOME}/MI325_cluster/nodes.txt"
DEFAULT_HOST_PATH="$USER_HOME"

# LangSplat-specific defaults
HOME_DIR="${USER_HOME}/LangSplat"
DEFAULT_DATASET_PATH="${HOME_DIR}/lerf_ovs/figurines"
DEFAULT_MODEL_PATH="output/figurines_ddp"
DEFAULT_CHECKPOINT="${HOME_DIR}/lerf_ovs/figurines/output/figurines_-1/chkpnt30000.pth"
DEFAULT_FEATURE_LEVELS="1 2 3"
DEFAULT_ITERATIONS=30000

# ============================================================
# Parse command line arguments
# ============================================================
NODES=""
NODES_FILE=""
NUM_NODES=""
SSH_USER=""
SSH_KEY=""
GPUS_PER_NODE=${GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
DOCKER_IMAGE=${DOCKER_IMAGE:-$DEFAULT_DOCKER_IMAGE}
HOST_PATH=${HOST_PATH:-$DEFAULT_HOST_PATH}
DATASET_PATH=${DATASET_PATH:-$DEFAULT_DATASET_PATH}
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
FEATURE_LEVELS=${FEATURE_LEVELS:-$DEFAULT_FEATURE_LEVELS}
ITERATIONS=${ITERATIONS:-$DEFAULT_ITERATIONS}
DRY_RUN=0
INSIDE_CONTAINER=0
SKIP_PIP_INSTALL=0
SKIP_CLEANUP=0
SKIP_RENDER=0

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "LangSplat Bare Metal Multi-Node DDP Training Launcher"
    echo ""
    echo "Options for SSH-based launch (from any node, outside container):"
    echo "  --num-nodes <n>        Number of nodes to use (reads from nodes file)"
    echo "  --nodes-file <path>    Path to file with node list, one per line"
    echo "                         (default: ~/MI325_cluster/nodes.txt)"
    echo "  --nodes <list>         Comma-separated list of node hostnames/IPs"
    echo "                         (alternative to --num-nodes + --nodes-file)"
    echo "  --user <username>      SSH username (default: current user)"
    echo "  --key <path>           Path to SSH private key (optional)"
    echo "  --gpus-per-node <n>    Number of GPUs per node (default: 8)"
    echo "  --docker-image <img>   Docker image to use (default: $DEFAULT_DOCKER_IMAGE)"
    echo "  --host-path <path>     Host path to mount in container (default: $DEFAULT_HOST_PATH)"
    echo "  --skip-pip-install     Skip pip package installation"
    echo "  --skip-cleanup         Skip cleanup of existing processes before launch"
    echo "  --dry-run              Print commands without executing"
    echo ""
    echo "Options for training configuration:"
    echo "  --dataset-path <path>  Path to the dataset (default: $DEFAULT_DATASET_PATH)"
    echo "  --model-path <path>    Model output path (default: $DEFAULT_MODEL_PATH)"
    echo "  --checkpoint <path>    Path to RGB checkpoint (default: $DEFAULT_CHECKPOINT)"
    echo "  --feature-levels <l>   Space-separated feature levels (default: '$DEFAULT_FEATURE_LEVELS')"
    echo "  --iterations <n>       Training iterations per level (default: $DEFAULT_ITERATIONS)"
    echo "  --skip-render          Skip render step after training each level"
    echo ""
    echo "Options for inside-container execution:"
    echo "  --inside-container     Run training directly (already inside container)"
    echo ""
    echo "Environment variables for manual launch (on each node):"
    echo "  MASTER_ADDR            IP address of master node"
    echo "  MASTER_PORT            Port for communication (default: 29500)"
    echo "  NNODES                 Total number of nodes"
    echo "  NODE_RANK              Rank of this node (0 = master)"
    echo "  GPUS_PER_NODE          GPUs per node (default: 8)"
    echo ""
    echo "Examples:"
    echo "  # 2-node training (first 2 nodes from nodes.txt):"
    echo "  $0 --num-nodes 2"
    echo ""
    echo "  # 4-node training with custom dataset:"
    echo "  $0 --num-nodes 4 --dataset-path /path/to/data"
    echo ""
    echo "  # Skip pip install (if already installed):"
    echo "  $0 --num-nodes 2 --skip-pip-install"
    echo ""
    echo "  # Train only feature level 1:"
    echo "  $0 --num-nodes 2 --feature-levels '1'"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --nodes-file)
            NODES_FILE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --user)
            SSH_USER="$2"
            shift 2
            ;;
        --key)
            SSH_KEY="$2"
            shift 2
            ;;
        --gpus-per-node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --host-path)
            HOST_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --feature-levels)
            FEATURE_LEVELS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --skip-pip-install)
            SKIP_PIP_INSTALL=1
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=1
            shift
            ;;
        --skip-render)
            SKIP_RENDER=1
            shift
            ;;
        --inside-container)
            INSIDE_CONTAINER=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# ============================================================
# If --num-nodes is specified, read nodes from file
# ============================================================
if [ -n "$NUM_NODES" ]; then
    NODES_FILE="${NODES_FILE:-$DEFAULT_NODES_FILE}"

    if [ ! -f "$NODES_FILE" ]; then
        echo "Error: Nodes file not found: $NODES_FILE"
        echo "Please create the file or specify --nodes-file"
        exit 1
    fi

    # Read nodes from file (skip empty lines and comments)
    mapfile -t ALL_NODES < <(grep -v '^#' "$NODES_FILE" | grep -v '^[[:space:]]*$' | head -n "$NUM_NODES")

    AVAILABLE_NODES=${#ALL_NODES[@]}
    if [ "$AVAILABLE_NODES" -lt "$NUM_NODES" ]; then
        echo "Warning: Requested $NUM_NODES nodes but only $AVAILABLE_NODES available in $NODES_FILE"
        echo "Using $AVAILABLE_NODES nodes"
    fi

    if [ "$AVAILABLE_NODES" -eq 0 ]; then
        echo "Error: No valid nodes found in $NODES_FILE"
        exit 1
    fi

    # Convert array to comma-separated string
    NODES=$(IFS=','; echo "${ALL_NODES[*]}")
    echo "Read ${#ALL_NODES[@]} node(s) from $NODES_FILE: $NODES"
fi

# ============================================================
# Function to get IP address of current machine
# ============================================================
get_local_ip() {
    ip route get 1 2>/dev/null | awk '{print $7; exit}' || \
    hostname -I 2>/dev/null | awk '{print $1}' || \
    hostname -i 2>/dev/null | awk '{print $1}' || \
    echo "127.0.0.1"
}

# ============================================================
# SSH-based launch mode (launches Docker on each node)
# ============================================================
if [ -n "$NODES" ]; then
    echo "========================================"
    echo "LangSplat Bare Metal Multi-Node Launcher (SSH + Docker Mode)"
    echo "========================================"

    # Parse node list
    IFS=',' read -ra NODE_ARRAY <<< "$NODES"
    NNODES=${#NODE_ARRAY[@]}

    if [ $NNODES -lt 1 ]; then
        echo "Error: No nodes specified"
        exit 1
    fi

    # First node is master
    MASTER_NODE="${NODE_ARRAY[0]}"
    MASTER_ADDR="$MASTER_NODE"
    MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}

    # Set SSH options
    SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if [ -n "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    if [ -n "$SSH_USER" ]; then
        SSH_PREFIX="${SSH_USER}@"
    else
        SSH_PREFIX=""
    fi

    # Working directory inside container
    WORK_DIR="${HOME_DIR}"

    echo "Master node:     $MASTER_NODE"
    echo "Master port:     $MASTER_PORT"
    echo "Total nodes:     $NNODES"
    echo "GPUs per node:   $GPUS_PER_NODE"
    echo "Docker image:    $DOCKER_IMAGE"
    echo "Host path:       $HOST_PATH"
    echo "Work dir:        $WORK_DIR"
    echo "Dataset path:    $DATASET_PATH"
    echo "Model path:      $MODEL_PATH"
    echo "Checkpoint:      $CHECKPOINT"
    echo "Feature levels:  $FEATURE_LEVELS"
    echo "Iterations:      $ITERATIONS"
    echo "Skip pip install: $SKIP_PIP_INSTALL"
    echo "Skip render:     $SKIP_RENDER"
    echo "Log directory:   $LOG_DIR"
    echo "========================================"

    # Generate unique job ID based on timestamp
    JOB_ID="langsplat_$(date +%Y%m%d%H%M%S)"

    # Create log directory for this job
    LOG_DIR="${HOME_DIR}/logs/${JOB_ID}"
    mkdir -p "$LOG_DIR"

    # Build pip install command
    if [ $SKIP_PIP_INSTALL -eq 1 ]; then
        PIP_CMD="echo 'Skipping pip install'"
    else
        PIP_CMD="pip install torchvision==0.25.0+rocm7.1 --index-url https://download.pytorch.org/whl/rocm7.1 && \
pip install open-clip-torch plyfile jaxtyping typing pathlib && \
pip install submodules/segment-anything-langsplat --no-build-isolation && \
pip install --no-build-isolation git+https://github.com/ROCm/gsplat.git && \
pip install --no-build-isolation git+https://github.com/amd-wangfan/simple-knn.git@hip_support && \
pip install opencv-python"
    fi

    # ============================================================
    # Cleanup on all nodes before starting
    # ============================================================
    if [ $SKIP_CLEANUP -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Cleaning up ALL existing processes on all nodes..."
        echo "========================================"

        for NODE in "${NODE_ARRAY[@]}"; do
            echo "Cleaning up node: $NODE"

            CLEANUP_CMD="
                echo '=== Comprehensive Cleanup on $NODE ===' && \

                echo '--- Stopping ALL running Docker containers ---' && \
                docker ps -q | xargs -r docker stop 2>/dev/null || true && \
                docker ps -aq | xargs -r docker rm -f 2>/dev/null || true && \

                echo '--- Killing Python/PyTorch processes (excluding Cursor) ---' && \
                for pid in \$(pgrep -f 'python' 2>/dev/null); do \
                    if ! grep -q 'cursor' /proc/\$pid/cmdline 2>/dev/null; then \
                        kill -9 \$pid 2>/dev/null || true; \
                    fi; \
                done && \
                for pid in \$(pgrep -f 'torchrun' 2>/dev/null); do \
                    if ! grep -q 'cursor' /proc/\$pid/cmdline 2>/dev/null; then \
                        kill -9 \$pid 2>/dev/null || true; \
                    fi; \
                done && \
                for pid in \$(pgrep -f 'torch.distributed' 2>/dev/null); do \
                    if ! grep -q 'cursor' /proc/\$pid/cmdline 2>/dev/null; then \
                        kill -9 \$pid 2>/dev/null || true; \
                    fi; \
                done && \

                echo '--- Killing processes using GPUs ---' && \
                if command -v rocm-smi &>/dev/null; then \
                    for pid in \$(rocm-smi --showpidgpus 2>/dev/null | grep -oP '\\d+' | sort -u); do \
                        echo \"Killing GPU process: \$pid\"; \
                        kill -9 \$pid 2>/dev/null || true; \
                    done; \
                fi && \

                echo '--- Killing any MPI processes ---' && \
                pkill -9 -f 'mpirun' 2>/dev/null || true && \
                pkill -9 -f 'mpiexec' 2>/dev/null || true && \
                pkill -9 -f 'orted' 2>/dev/null || true && \

                echo '--- Killing NCCL/RCCL related processes ---' && \
                pkill -9 -f 'nccl' 2>/dev/null || true && \
                pkill -9 -f 'rccl' 2>/dev/null || true && \

                echo '--- Clearing shared memory ---' && \
                rm -rf /dev/shm/* 2>/dev/null || true && \

                echo '--- Resetting AMD GPUs ---' && \
                if command -v rocm-smi &>/dev/null; then \
                    echo 'GPU status before cleanup:'; \
                    rocm-smi --showuse 2>/dev/null || true; \
                    echo 'Attempting GPU reset...'; \
                    rocm-smi --resetclocks 2>/dev/null || true; \
                fi && \

                echo '--- Final GPU memory status ---' && \
                if command -v rocm-smi &>/dev/null; then \
                    rocm-smi --showmeminfo vram 2>/dev/null | head -30 || true; \
                fi && \

                echo '=== Cleanup complete on $NODE ==='
            "

            if [ $DRY_RUN -eq 1 ]; then
                echo "[DRY RUN] ssh $SSH_OPTS ${SSH_PREFIX}${NODE} \"$CLEANUP_CMD\""
            else
                ssh $SSH_OPTS ${SSH_PREFIX}${NODE} "$CLEANUP_CMD" &
            fi
        done

        # Wait for all cleanup processes to complete
        if [ $DRY_RUN -eq 0 ]; then
            wait
            echo ""
            echo "========================================"
            echo "Cleanup complete on all nodes."
            echo "========================================"
            # Brief pause to ensure processes are fully terminated
            sleep 3
        fi
    else
        echo ""
        echo "Skipping cleanup (--skip-cleanup specified)"
    fi

    # ============================================================
    # Launch training on all nodes
    # ============================================================
    echo ""
    echo "========================================"
    echo "Launching LangSplat DDP training on all nodes..."
    echo "========================================"

    # IB driver install command
    IB_INSTALL_CMD='bash /home/douglas.jia@amd.com/set_ib.sh'

    PIDS=()
    for i in "${!NODE_ARRAY[@]}"; do
        NODE="${NODE_ARRAY[$i]}"
        NODE_RANK=$i
        CONTAINER_NAME="langsplat_${JOB_ID}_node${NODE_RANK}"

        echo ""
        echo "Launching on node $NODE (rank $NODE_RANK)..."

        # Build the docker run command
        # Uses --privileged for full device access (kfd, dri, infiniband)
        # Uses --tmpfs for /dev/shm with 200G (LangSplat needs large shared memory)
        DOCKER_CMD="docker run \
            --rm \
            --name=${CONTAINER_NAME} \
            --privileged \
            --network host \
            --cap-add=IPC_LOCK \
            --ipc=host \
            --ulimit memlock=-1:-1 \
            --volume /dev/infiniband:/dev/infiniband \
            --tmpfs /dev/shm:size=200G \
            -v ${HOST_PATH}:${HOST_PATH} \
            -v /home/primus/data/libbnxt:/home/primus/data/libbnxt \
            -w ${WORK_DIR} \
            -e MASTER_ADDR=${MASTER_ADDR} \
            -e MASTER_PORT=${MASTER_PORT} \
            -e NNODES=${NNODES} \
            -e NODE_RANK=${NODE_RANK} \
            -e GPUS_PER_NODE=${GPUS_PER_NODE} \
            -e DATASET_PATH=${DATASET_PATH} \
            -e MODEL_PATH=${MODEL_PATH} \
            -e CHECKPOINT=${CHECKPOINT} \
            -e FEATURE_LEVELS='${FEATURE_LEVELS}' \
            -e ITERATIONS=${ITERATIONS} \
            -e SKIP_RENDER=${SKIP_RENDER} \
            -e JOB_ID=${JOB_ID} \
            ${DOCKER_IMAGE} \
            bash -c '${IB_INSTALL_CMD} && ${PIP_CMD} && bash ./launch_baremetal.sh --inside-container'"

        # Per-node log file: <node>_rank<rank>.log
        LOG_FILE="${LOG_DIR}/${NODE}_rank${NODE_RANK}.log"

        if [ $DRY_RUN -eq 1 ]; then
            echo "[DRY RUN] ssh $SSH_OPTS ${SSH_PREFIX}${NODE} \"$DOCKER_CMD\""
            echo "[DRY RUN] Log would be saved to: ${LOG_FILE}"
        else
            # Launch in background, tee output to both terminal and log file
            ssh $SSH_OPTS ${SSH_PREFIX}${NODE} "$DOCKER_CMD" 2>&1 | tee "${LOG_FILE}" &
            PIDS+=($!)
            echo "Started on $NODE with PID ${PIDS[-1]}, container: $CONTAINER_NAME"
            echo "  Log file: ${LOG_FILE}"
        fi
    done

    if [ $DRY_RUN -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Waiting for all nodes to complete..."
        echo "PIDs: ${PIDS[*]}"
        echo "Log directory: ${LOG_DIR}"
        echo "Press Ctrl+C to stop all containers"
        echo "========================================"

        # Trap to handle Ctrl+C - stop all containers
        trap 'echo "Stopping containers..."; for i in "${!NODE_ARRAY[@]}"; do NODE="${NODE_ARRAY[$i]}"; CONTAINER_NAME="langsplat_${JOB_ID}_node${i}"; ssh $SSH_OPTS ${SSH_PREFIX}${NODE} "docker stop $CONTAINER_NAME 2>/dev/null || true" & done; wait; exit 130' INT

        # Wait for all processes
        FAILED=0
        for i in "${!PIDS[@]}"; do
            PID=${PIDS[$i]}
            NODE="${NODE_ARRAY[$i]}"
            if wait $PID; then
                echo "Node $NODE (PID $PID) completed successfully"
            else
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 130 ]; then
                    echo "Node $NODE (PID $PID) was interrupted"
                else
                    echo "Node $NODE (PID $PID) failed with exit code $EXIT_CODE"
                    FAILED=1
                fi
            fi
        done

        if [ $FAILED -eq 1 ]; then
            echo "Some nodes failed. Check logs in: ${LOG_DIR}"
            exit 1
        fi

        echo ""
        echo "All nodes completed successfully!"
        echo "Logs saved to: ${LOG_DIR}"
        ls -lh "${LOG_DIR}/"
    fi

    exit 0
fi

# ============================================================
# Inside-container mode or manual launch mode
# ============================================================
if [ $INSIDE_CONTAINER -eq 1 ]; then
    echo "========================================"
    echo "LangSplat DDP Training (Inside Container)"
    echo "========================================"
else
    echo "========================================"
    echo "LangSplat DDP Training (Manual Mode)"
    echo "========================================"
    echo "Note: Use --inside-container if already in Docker"
    echo "      Or use --num-nodes to launch via SSH+Docker"
fi

# Determine master address
if [ -z "$MASTER_ADDR" ]; then
    if [ "${NODE_RANK:-0}" -eq 0 ]; then
        # We are master, use our own IP
        MASTER_ADDR=$(get_local_ip)
        echo "Auto-detected master address: $MASTER_ADDR"
    else
        echo "Error: MASTER_ADDR must be set for worker nodes (NODE_RANK > 0)"
        echo ""
        print_usage
        exit 1
    fi
fi

# Set defaults
MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}

# ============================================================
# Setup logging - all output goes to both terminal and log file
# ============================================================
NODE_NAME=${HOSTNAME:-$(hostname)}
JOB_ID=${JOB_ID:-"langsplat_$(date +%Y%m%d%H%M%S)"}
LOG_DIR="${HOME_DIR}/logs/${JOB_ID}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${NODE_NAME}_rank${NODE_RANK}.log"
echo "Log directory: $LOG_DIR"
echo "Log file:      $LOG_FILE"

# Redirect all subsequent output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# ROCm/HIP environment variables
# ============================================================
export HIP_FORCE_DEV_KERNARG=1
export MIOPEN_FIND_MODE=3
export MIOPEN_FIND_ENFORCE=3
export ROCBLAS_USE_HIPBLASLT=1
export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0
export HIP_LAUNCH_BLOCKING=0
export HSA_KERNARG_POOL_SIZE=12582912

# MIOpen cache directories (node-specific to avoid conflicts)
export MIOPEN_USER_DB_PATH=/tmp/langsplat_${USER:-user}_node${NODE_RANK}/.config
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/langsplat_${USER:-user}_node${NODE_RANK}/.cache
mkdir -p ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_CUSTOM_CACHE_DIR}

# Clear torch extensions cache to avoid stale builds
rm -rf ~/.cache/torch_extensions

# ============================================================
# RCCL/NCCL parameters for multi-node communication
# ============================================================
export NCCL_DEBUG=INFO
# 4 NICs per node - adjust based on your InfiniBand/RoCE configuration
export NCCL_IB_HCA="rdma0:1,rdma2:1,rdma4:1,rdma6:1"

# Torch Distributed Networking
# Use POD_IP for distributed hostname if available (Kubernetes)
if [ -n "${POD_IP}" ]; then
    export TORCH_DISTRIBUTED_HOSTNAME=${POD_IP}
    echo "Using POD_IP for TORCH_DISTRIBUTED_HOSTNAME: ${POD_IP}"
fi

# GPU visibility
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ============================================================
# Print configuration
# ============================================================
echo "Master Address:   $MASTER_ADDR"
echo "Master Port:      $MASTER_PORT"
echo "Number of Nodes:  $NNODES"
echo "Node Rank:        $NODE_RANK"
echo "GPUs per Node:    $GPUS_PER_NODE"
echo "Dataset Path:     $DATASET_PATH"
echo "Model Path:       $MODEL_PATH"
echo "Checkpoint:       $CHECKPOINT"
echo "Feature Levels:   $FEATURE_LEVELS"
echo "Iterations:       $ITERATIONS"
echo "Skip Render:      $SKIP_RENDER"
echo "NCCL_IB_HCA:      $NCCL_IB_HCA"
echo "MIOPEN_USER_DB:   $MIOPEN_USER_DB_PATH"
echo "Log file:         $LOG_FILE"
echo "========================================"

# ============================================================
# Training loop: train and render each feature level
# ============================================================
TOTAL_LEVELS=$(echo $FEATURE_LEVELS | wc -w)
CURRENT=0

for level in $FEATURE_LEVELS; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================"
    echo "Feature Level $level ($CURRENT/$TOTAL_LEVELS)"
    echo "========================================"

    # --------------------------------------------------------
    # Step 1: DDP Training (all nodes participate via torchrun)
    # --------------------------------------------------------
    echo "[Level $level] Starting DDP training on all ${NNODES} node(s), ${GPUS_PER_NODE} GPUs each..."

    OMP_NUM_THREADS=1 torchrun \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_ddp.py \
        -s "$DATASET_PATH" \
        -m "$MODEL_PATH" \
        --start_checkpoint "$CHECKPOINT" \
        --feature_level $level \
        --iterations $ITERATIONS \
        --save_iterations 7000 $ITERATIONS \
        --checkpoint_iterations 7000 $ITERATIONS \
        --test_iterations 7000 $ITERATIONS

    echo "[Level $level] DDP training complete on node $NODE_RANK."

    # --------------------------------------------------------
    # Step 2: Render (master node only, single-GPU inference)
    # --------------------------------------------------------
    if [ "$SKIP_RENDER" -eq 1 ]; then
        echo "[Level $level] Skipping render (--skip-render specified)."
    elif [ "$NODE_RANK" -eq 0 ]; then
        echo "[Level $level] Running render on master node..."

        python render.py \
            -s "$DATASET_PATH" \
            -m "${MODEL_PATH}_${level}" \
            --feature_level ${level} \
            --include_feature

        echo "[Level $level] Render complete."
    else
        echo "[Level $level] Worker node $NODE_RANK - skipping render (master only)."
    fi

    echo "[Level $level] Done."
done

echo ""
echo "========================================"
echo "LangSplat DDP training complete on node $NODE_RANK!"
echo "All feature levels ($FEATURE_LEVELS) processed."
echo "Log saved to: $LOG_FILE"
echo "========================================"
