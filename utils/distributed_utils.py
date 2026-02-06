#
# Distributed training utilities for LangSplat language feature training
#

import os
import torch
import torch.distributed as dist
from typing import Tuple


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """
    Initialize distributed training environment.
    
    Returns:
        rank: Global rank of this process
        local_rank: Local rank on this node (used for GPU assignment)
        world_size: Total number of processes
    """
    # Check if we're running in a distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Not distributed - single GPU mode
        return 0, 0, 1
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the world size (total number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """
    Reduce loss across all processes for logging.
    Returns the mean loss across all ranks.
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return loss
    
    reduced_loss = loss.clone()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss /= get_world_size()
    return reduced_loss


class DistributedCameraSampler:
    """
    Distributes cameras across processes for training.
    
    Each process gets a different subset of cameras per epoch.
    By default, the last uneven cameras are dropped so all ranks
    process the same number of cameras per epoch.
    """
    
    def __init__(
        self,
        cameras: list,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        """
        Args:
            cameras: List of camera objects
            rank: Rank of the current process
            world_size: Total number of processes
            shuffle: Whether to shuffle cameras
            seed: Random seed for reproducibility
            drop_last: Drop the last uneven cameras each epoch
        """
        self.cameras = cameras
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._indices = None
        self._current_idx = 0
        self._refresh_indices()
    
    def _refresh_indices(self):
        """Refresh the camera indices for the current epoch."""
        # Create a deterministic ordering based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            indices = torch.randperm(len(self.cameras), generator=g).tolist()
        else:
            indices = list(range(len(self.cameras)))

        if self.drop_last:
            if len(indices) < self.world_size:
                raise ValueError(
                    "Number of cameras is less than world_size with drop_last=True. "
                    "Reduce world_size or disable drop_last."
                )
            total_size = len(indices) - (len(indices) % self.world_size)
            indices = indices[:total_size]
        
        # Each rank gets every world_size-th element, starting from rank
        self._indices = indices[self.rank::self.world_size]
        self._current_idx = 0
    
    def get_camera(self) -> object:
        """Get the next camera for this process."""
        if self._current_idx >= len(self._indices):
            self.epoch += 1
            self._refresh_indices()
        
        camera = self.cameras[self._indices[self._current_idx]]
        self._current_idx += 1
        return camera
    
    def set_epoch(self, epoch: int):
        """Set the epoch (for reproducibility across restarts)."""
        self.epoch = epoch
        self._refresh_indices()
    
    def __len__(self):
        """Return the number of cameras this rank will process per epoch."""
        return len(self._indices)


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
