#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# DDP (Distributed Data Parallel) training for LangSplat language feature training
# NOTE: This script is ONLY for language feature training (--include_feature).
#       For RGB/geometry training, use the original train.py

import os
import torch
import torch.distributed as dist
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    reduce_loss,
    DistributedCameraSampler,
    print_rank0,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_ddp(dataset, opt, pipe, testing_iterations, saving_iterations, 
                 checkpoint_iterations, checkpoint, debug_from,
                 rank, local_rank, world_size, epochs=None):
    """
    DDP training function for LangSplat language feature training.
    
    This is specifically for language feature training where:
    - Only _language_feature parameter is trained
    - All geometry parameters (xyz, scaling, rotation, etc.) are frozen
    - No densification occurs - model structure is fixed
    
    Key DDP features:
    1. Each rank processes different cameras in parallel
    2. Gradients for _language_feature are averaged across ranks
    3. Only rank 0 handles logging, checkpointing, and tensorboard
    """
    
    # Ensure this is language feature training
    if not opt.include_feature:
        raise ValueError(
            "train_ddp.py is only for language feature training (--include_feature).\n"
            "For RGB/geometry training, use the original train.py"
        )
    
    first_iter = 0
    
    # Only rank 0 sets up output directory and tensorboard
    if is_main_process():
        tb_writer = prepare_output_and_logger(dataset)
    else:
        tb_writer = None
    
    # Wait for rank 0 to create output directory
    barrier()
    
    # Set device for this rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize model and scene
    gaussians = GaussianModel(dataset.sh_degree, device=str(device))
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Language feature training requires a checkpoint from RGB training
    if not checkpoint:
        raise ValueError(
            "Language feature training requires a checkpoint from RGB training.\n"
            "Please provide --start_checkpoint path/to/rgb_checkpoint.pth"
        )
    
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False, map_location=device)
    # If loading from RGB checkpoint, reset iteration counter
    if len(model_params) == 12:
        first_iter = 0
    gaussians.restore(model_params, opt)
    
    # Background color tensor on the correct device
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    
    # Timing events
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    
    # Create distributed camera sampler (drop last uneven cameras each epoch)
    train_cameras = scene.getTrainCameras().copy()
    num_cameras = len(train_cameras)
    if num_cameras < world_size:
        raise ValueError(
            f"Number of cameras ({num_cameras}) is less than world_size ({world_size}). "
            "Reduce world_size to use drop_last sampling."
        )
    camera_sampler = DistributedCameraSampler(
        cameras=train_cameras,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    
    # Calculate iterations based on epochs if specified
    # 1 epoch = all cameras seen once across all GPUs (dropping last uneven cameras)
    cameras_per_iteration = world_size  # Each GPU processes 1 camera per iteration
    iterations_per_epoch = num_cameras // cameras_per_iteration
    if iterations_per_epoch == 0:
        raise ValueError(
            f"iterations_per_epoch is 0 with {num_cameras} cameras and world_size {world_size}. "
            "Reduce world_size to proceed."
        )
    
    if epochs is not None:
        total_iterations = epochs * iterations_per_epoch
        opt.iterations = total_iterations
        if is_main_process():
            print(f"\n=== Epoch-based Training ===")
            print(f"Epochs requested: {epochs}")
            print(f"Iterations per epoch: {iterations_per_epoch}")
            print(f"Total iterations: {total_iterations}")
    
    # Print training statistics
    if is_main_process():
        print(f"\n=== Training Statistics ===")
        print(f"Number of training cameras: {num_cameras}")
        print(f"World size (GPUs): {world_size}")
        print(f"Cameras processed per iteration: {cameras_per_iteration}")
        print(f"Iterations per epoch: {iterations_per_epoch}")
        print(f"Total iterations: {opt.iterations}")
        print(f"Total epochs: {opt.iterations / iterations_per_epoch:.2f}")
        print(f"Total camera views to process: {opt.iterations * cameras_per_iteration}")
        print("=" * 30 + "\n")
    
    ema_loss_for_log = 0.0
    
    # Progress bar only on rank 0
    if is_main_process():
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    first_iter += 1
    current_epoch = (first_iter - 1) // iterations_per_epoch
    camera_sampler.set_epoch(current_epoch)
    
    # Debug timing accumulators
    import time
    debug_times = {
        'render': 0.0, 'io': 0.0, 'loss': 0.0, 'backward': 0.0,
        'sync1': 0.0, 'allreduce': 0.0, 'sync2': 0.0, 
        'optimizer': 0.0, 'barrier': 0.0, 'total': 0.0
    }
    debug_count = 0
    
    for iteration in range(first_iter, opt.iterations + 1):
        # Track epoch changes
        new_epoch = (iteration - 1) // iterations_per_epoch
        if new_epoch > current_epoch:
            current_epoch = new_epoch
            camera_sampler.set_epoch(current_epoch)
            if is_main_process():
                print(f"\n[Epoch {new_epoch}] Starting...")
        
        # Debug timing
        t_start = time.perf_counter()
        
        # Ensure clean timing - sync before recording start
        torch.cuda.synchronize()
        iter_start.record()
        
        # Update learning rate (no-op for language feature training, but kept for consistency)
        gaussians.update_learning_rate(iteration)
        
        # Pick a camera using distributed sampler (each rank gets different cameras)
        viewpoint_cam = camera_sampler.get_camera()
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        t0 = time.perf_counter()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt)
        language_feature = render_pkg["language_feature_image"]
        torch.cuda.synchronize()  # Ensure render complete for timing
        t1 = time.perf_counter()
        debug_times['render'] += t1 - t0
        
        # Loss computation - L1 loss on language features
        t0 = time.perf_counter()
        gt_language_feature, language_feature_mask = viewpoint_cam.get_language_feature(
            language_feature_dir=dataset.lf_path, 
            feature_level=dataset.feature_level
        )
        t1 = time.perf_counter()
        debug_times['io'] += t1 - t0
        
        t0 = time.perf_counter()
        Ll1 = l1_loss(language_feature * language_feature_mask, gt_language_feature * language_feature_mask)
        loss = Ll1
        t1 = time.perf_counter()
        debug_times['loss'] += t1 - t0
        
        # Backward pass
        t0 = time.perf_counter()
        loss.backward()
        t1 = time.perf_counter()
        debug_times['backward'] += t1 - t0
        
        # CRITICAL: Synchronize CUDA before gradient communication
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        debug_times['sync1'] += t1 - t0
        
        with torch.no_grad():
            # Average gradients for _language_feature across all ranks
            t0 = time.perf_counter()
            if world_size > 1:
                if gaussians._language_feature.grad is not None:
                    dist.all_reduce(gaussians._language_feature.grad, op=dist.ReduceOp.SUM)
                    gaussians._language_feature.grad.div_(world_size)
                else:
                    if is_main_process():
                        print(f"WARNING: _language_feature.grad is None at iteration {iteration}")
            t1 = time.perf_counter()
            debug_times['allreduce'] += t1 - t0
            
            # Synchronize after all_reduce
            t0 = time.perf_counter()
            if world_size > 1:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            debug_times['sync2'] += t1 - t0
            
            # Record timing AFTER all distributed operations
            iter_end.record()
            torch.cuda.synchronize()
            
            # Reduce loss for logging
            loss_for_log = reduce_loss(loss.detach()).item()
            ema_loss_for_log = 0.4 * loss_for_log + 0.6 * ema_loss_for_log
            
            # Progress bar update (rank 0 only)
            if is_main_process():
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
            
            # Log and save (rank 0 only)
            if is_main_process():
                training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), 
                               testing_iterations, scene, render, (pipe, background, opt))
                if iteration in saving_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    scene.save(iteration)
            
            # Optimizer step
            t0 = time.perf_counter()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            t1 = time.perf_counter()
            debug_times['optimizer'] += t1 - t0
            
            # Checkpoint saving (rank 0 only)
            if is_main_process() and iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(opt.include_feature), iteration), 
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )
            
            # Barrier to keep all ranks in sync
            t0 = time.perf_counter()
            if world_size > 1:
                barrier()
            t1 = time.perf_counter()
            debug_times['barrier'] += t1 - t0
        
        debug_times['total'] += time.perf_counter() - t_start
        debug_count += 1
        
        # Print debug timing every 100 iterations
        if iteration % 100 == 0:
            io_sum = torch.tensor(debug_times['io'], device=device)
            io_max = torch.tensor(debug_times['io'], device=device)
            if world_size > 1:
                dist.all_reduce(io_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(io_max, op=dist.ReduceOp.MAX)

            if is_main_process():
                print(f"\n=== DEBUG TIMING (avg over {debug_count} iters) ===")
                for k, v in debug_times.items():
                    avg_ms = (v / debug_count) * 1000
                    print(f"  {k}: {avg_ms:.2f} ms")
                io_avg_ms = (io_sum.item() / (world_size * debug_count)) * 1000
                io_max_ms = (io_max.item() / debug_count) * 1000
                print(f"  io (avg per-rank): {io_avg_ms:.2f} ms")
                print(f"  io (max per-rank): {io_max_ms:.2f} ms")
                print(f"  Effective it/s: {debug_count / debug_times['total']:.2f}")
                print("=" * 45)


def prepare_output_and_logger(args):
    """Prepare output directory and tensorboard logger (rank 0 only)."""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, 
                   scene: Scene, renderFunc, renderArgs):
    """Training report for language feature training (rank 0 only)."""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
    
    # Validation at specified iterations
    if iteration in testing_iterations:
        print(f'Validation at iter {iteration}')
        torch.cuda.empty_cache()
        
        # For language feature training, we just log the training loss
        # Full evaluation would require language feature ground truth for test set
        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="DDP Training script for LangSplat language feature training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    # DDP specific arguments
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend (nccl, gloo)")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Train for N epochs instead of iterations. 1 epoch = all cameras seen once across all GPUs. "
                             "If specified, overrides --iterations.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale learning rate by world_size (recommended for DDP with larger effective batch)")
    parser.add_argument("--match_single_gpu_updates", action="store_true", default=False,
                        help="Automatically set iterations to match single-GPU training updates (iterations * world_size)")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Force include_feature=True for this script
    args.include_feature = True
    
    # Initialize distributed training
    rank, local_rank, world_size = setup_distributed(backend=args.dist_backend)
    
    # Handle --match_single_gpu_updates: multiply iterations by world_size
    # This ensures same number of gradient updates as single GPU training
    if args.match_single_gpu_updates and args.epochs is None:
        original_iterations = args.iterations
        args.iterations = args.iterations * world_size
        if is_main_process():
            print(f"[match_single_gpu_updates] Scaling iterations: {original_iterations} -> {args.iterations}")
    
    # Handle --scale_lr: scale learning rate by world_size
    # Linear scaling rule for larger effective batch size
    if args.scale_lr:
        original_lr = args.language_feature_lr
        args.language_feature_lr = args.language_feature_lr * world_size
        if is_main_process():
            print(f"[scale_lr] Scaling learning rate: {original_lr} -> {args.language_feature_lr}")
    
    if args.epochs is None:
        args.save_iterations.append(args.iterations)
    
    # Modify model path to include feature level
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    
    if is_main_process():
        print("=" * 60)
        print("LangSplat DDP Training - Language Feature Training")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Rank: {rank}, Local rank: {local_rank}")
        print(f"Model path: {args.model_path}")
        print(f"Feature level: {args.feature_level}")
        print(f"Learning rate: {args.language_feature_lr}")
        print(f"Iterations: {args.iterations}")
        print(args)
        print("=" * 60)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Set random seed per rank for reproducibility but different sampling
    torch.manual_seed(42 + rank)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Run DDP training
    training_ddp(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from,
        rank,
        local_rank,
        world_size,
        epochs=args.epochs
    )
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process():
        print("\nDDP Training complete.")
