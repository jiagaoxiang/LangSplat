#!/usr/bin/env python
"""
3DGS RGB Quality Comparison Script
Renders images from the trained 3DGS model and compares with ground truth.

Usage:
    # Render and compare all training images
    python compare_rgb_quality.py -s lerf_ovs/figurines -m lerf_ovs/figurines/output/figurines_-1
    
    # Only render specific frames
    python compare_rgb_quality.py -s lerf_ovs/figurines -m lerf_ovs/figurines/output/figurines_-1 --frame_indices 0 50 100 150
    
    # Skip rendering if already done, just create comparison
    python compare_rgb_quality.py -s lerf_ovs/figurines -m lerf_ovs/figurines/output/figurines_-1 --skip_render
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import ssim
from utils.image_utils import psnr


def compute_metrics(rendered, gt):
    """Compute PSNR and SSIM between rendered and ground truth images."""
    # Ensure tensors are on the same device
    rendered = rendered.cuda()
    gt = gt.cuda()
    
    # PSNR - take mean across channels
    psnr_val = psnr(rendered, gt).mean().item()
    
    # SSIM
    ssim_val = ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).mean().item()
    
    return psnr_val, ssim_val


def create_comparison_figure(rendered, gt, frame_idx, psnr_val, ssim_val, save_path):
    """Create a side-by-side comparison figure with difference map."""
    # Convert to numpy (H, W, C)
    rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    
    # Compute difference
    diff = np.abs(rendered_np - gt_np)
    diff_gray = np.mean(diff, axis=2)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Ground Truth
    axes[0].imshow(np.clip(gt_np, 0, 1))
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # Rendered
    axes[1].imshow(np.clip(rendered_np, 0, 1))
    axes[1].set_title('Rendered (3DGS)', fontsize=14)
    axes[1].axis('off')
    
    # Difference (color)
    axes[2].imshow(np.clip(diff * 5, 0, 1))  # Amplify for visibility
    axes[2].set_title('Difference (5x amplified)', fontsize=14)
    axes[2].axis('off')
    
    # Difference heatmap
    im = axes[3].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Error Heatmap', fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Add metrics
    fig.suptitle(f'Frame {frame_idx:05d} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_and_compare(args):
    """Main function to render and compare images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup paths
    source_path = args.source_path
    model_path = args.model_path
    checkpoint_path = os.path.join(model_path, 'chkpnt30000.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Source path: {source_path}")
    print(f"Model path: {model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    print("\nLoading 3DGS model...")
    
    # Create a simple args namespace for model params
    class SimpleArgs:
        def __init__(self):
            self.sh_degree = 3
            self.source_path = source_path
            self.model_path = model_path
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.include_feature = False
            self.feature_level = -1
            self.language_features_name = "language_features_dim3"
            self.lf_path = os.path.join(source_path, "language_features_dim3")
    
    dataset_args = SimpleArgs()
    
    # Initialize Gaussian model
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, shuffle=False)
    
    # Load checkpoint
    print("Loading checkpoint...")
    model_params, _ = torch.load(checkpoint_path, weights_only=False)
    
    # Manually load model parameters (avoiding restore() optimizer issue in test mode)
    if len(model_params) == 12:
        # RGB checkpoint (no language features)
        (gaussians.active_sh_degree, 
         gaussians._xyz, 
         gaussians._features_dc, 
         gaussians._features_rest,
         gaussians._scaling, 
         gaussians._rotation, 
         gaussians._opacity,
         gaussians.max_radii2D, 
         _, _, _, 
         gaussians.spatial_lr_scale) = model_params
    elif len(model_params) == 13:
        # Language feature checkpoint
        (gaussians.active_sh_degree, 
         gaussians._xyz, 
         gaussians._features_dc, 
         gaussians._features_rest,
         gaussians._scaling, 
         gaussians._rotation, 
         gaussians._opacity,
         gaussians._language_feature,
         gaussians.max_radii2D, 
         _, _, _, 
         gaussians.spatial_lr_scale) = model_params
    else:
        raise ValueError(f"Unknown checkpoint format with {len(model_params)} parameters")
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Setup rendering
    bg_color = [0, 0, 0]  # Black background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    class PipeArgs:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False
    
    class RenderOptArgs:
        include_feature = False
    
    pipe_args = PipeArgs()
    render_opt_args = RenderOptArgs()
    
    # Get cameras
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    
    print(f"\nFound {len(train_cameras)} training cameras")
    print(f"Found {len(test_cameras)} test cameras")
    
    # Determine which frames to process
    if args.frame_indices:
        indices = args.frame_indices
    else:
        indices = list(range(len(train_cameras)))
    
    if args.max_frames:
        indices = indices[:args.max_frames]
    
    print(f"Processing {len(indices)} frames...")
    
    # Render and compare
    all_psnr = []
    all_ssim = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Rendering"):
            if idx >= len(train_cameras):
                print(f"Warning: Frame index {idx} out of range")
                continue
            
            camera = train_cameras[idx]
            
            # Render
            output = render(camera, gaussians, pipe_args, background, render_opt_args)
            rendered = output["render"]
            
            # Get ground truth
            gt = camera.original_image[0:3, :, :].cuda()
            
            # Compute metrics
            psnr_val, ssim_val = compute_metrics(rendered, gt)
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            
            # Create comparison figure
            save_path = output_dir / f"comparison_{idx:05d}.png"
            create_comparison_figure(rendered, gt, idx, psnr_val, ssim_val, save_path)
            
            # Also save individual images
            if args.save_individual:
                rendered_np = (rendered.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                cv2.imwrite(str(output_dir / f"rendered_{idx:05d}.png"), 
                           cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"gt_{idx:05d}.png"), 
                           cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))
    
    # Print summary statistics
    print("\n" + "="*60)
    print("QUALITY METRICS SUMMARY")
    print("="*60)
    print(f"Frames processed: {len(all_psnr)}")
    print(f"Average PSNR: {np.mean(all_psnr):.2f} dB (±{np.std(all_psnr):.2f})")
    print(f"Average SSIM: {np.mean(all_ssim):.4f} (±{np.std(all_ssim):.4f})")
    print(f"Min PSNR: {np.min(all_psnr):.2f} dB")
    print(f"Max PSNR: {np.max(all_psnr):.2f} dB")
    print(f"Min SSIM: {np.min(all_ssim):.4f}")
    print(f"Max SSIM: {np.max(all_ssim):.4f}")
    print("="*60)
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Frame,PSNR,SSIM\n")
        for i, (p, s) in enumerate(zip(all_psnr, all_ssim)):
            f.write(f"{indices[i]},{p:.4f},{s:.4f}\n")
        f.write(f"\nAverage,{np.mean(all_psnr):.4f},{np.mean(all_ssim):.4f}\n")
        f.write(f"Std,{np.std(all_psnr):.4f},{np.std(all_ssim):.4f}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Create a summary figure with best/worst examples
    if len(all_psnr) >= 4:
        print("\nCreating summary figure...")
        create_summary_figure(indices, all_psnr, all_ssim, output_dir)


def create_summary_figure(indices, all_psnr, all_ssim, output_dir):
    """Create a summary figure showing best and worst results."""
    # Find best and worst by PSNR
    sorted_idx = np.argsort(all_psnr)
    worst_idx = sorted_idx[:2]
    best_idx = sorted_idx[-2:][::-1]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Load and display best results
    for i, idx in enumerate(best_idx):
        frame_idx = indices[idx]
        img_path = output_dir / f"comparison_{frame_idx:05d}.png"
        if img_path.exists():
            img = plt.imread(str(img_path))
            axes[0, i*2].imshow(img)
            axes[0, i*2].set_title(f'Best #{i+1}: Frame {frame_idx}\nPSNR={all_psnr[idx]:.2f}dB', fontsize=10)
            axes[0, i*2].axis('off')
            axes[0, i*2+1].axis('off')
    
    # Load and display worst results
    for i, idx in enumerate(worst_idx):
        frame_idx = indices[idx]
        img_path = output_dir / f"comparison_{frame_idx:05d}.png"
        if img_path.exists():
            img = plt.imread(str(img_path))
            axes[1, i*2].imshow(img)
            axes[1, i*2].set_title(f'Worst #{i+1}: Frame {frame_idx}\nPSNR={all_psnr[idx]:.2f}dB', fontsize=10)
            axes[1, i*2].axis('off')
            axes[1, i*2+1].axis('off')
    
    fig.suptitle(f'3DGS Quality Summary\nAvg PSNR: {np.mean(all_psnr):.2f}dB | Avg SSIM: {np.mean(all_ssim):.4f}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary figure saved to: {output_dir / 'summary.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS RGB Quality Comparison")
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="Path to dataset (e.g., lerf_ovs/figurines)")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="Path to trained model (e.g., lerf_ovs/figurines/output/figurines_-1)")
    parser.add_argument("--output_dir", type=str, default="rgb_comparison_output",
                        help="Output directory for comparisons")
    parser.add_argument("--frame_indices", nargs='+', type=int, default=None,
                        help="Specific frame indices to render")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual rendered and GT images")
    
    args = parser.parse_args()
    render_and_compare(args)
