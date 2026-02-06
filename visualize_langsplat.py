#!/usr/bin/env python
"""
LangSplat Visualization Script
Generates images with text queries and compares with ground truth.

Usage:
    python visualize_langsplat.py --dataset_name figurines --text_queries "robot" "waldo" "porcelain hand"
    
    # Or use ground truth labels automatically:
    python visualize_langsplat.py --dataset_name figurines --use_gt_labels
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))

from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork


def load_gt_annotations(json_folder, dataset_name):
    """Load ground truth annotations from JSON files."""
    json_path = os.path.join(json_folder, dataset_name)
    gt_json_paths = sorted(glob.glob(os.path.join(json_path, 'frame_*.json')))
    
    annotations = {}
    all_labels = set()
    
    for js_path in gt_json_paths:
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        frame_name = gt_data['info']['name']
        idx = int(frame_name.split('_')[-1].split('.jpg')[0]) - 1
        h, w = gt_data['info']['height'], gt_data['info']['width']
        
        frame_ann = {
            'frame_name': frame_name,
            'height': h,
            'width': w,
            'objects': []
        }
        
        for obj in gt_data['objects']:
            label = obj['category']
            bbox = obj['bbox']
            segmentation = obj['segmentation']
            
            # Create mask from polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
            
            frame_ann['objects'].append({
                'label': label,
                'bbox': bbox,
                'mask': mask
            })
            all_labels.add(label)
        
        annotations[idx] = frame_ann
    
    return annotations, list(all_labels)


def apply_colormap(value, cmap_name='turbo'):
    """Apply colormap to values in [0, 1]."""
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(value.cpu().numpy())
    return (colored[..., :3] * 255).astype(np.uint8)


def visualize_relevancy(relevancy_map, rgb_image, threshold=0.5):
    """Create visualization with relevancy overlay on RGB image."""
    # Normalize relevancy
    rel_norm = relevancy_map - relevancy_map.min()
    rel_norm = rel_norm / (rel_norm.max() + 1e-9)
    
    # Apply colormap
    heatmap = apply_colormap(rel_norm)
    
    # Create composite: show heatmap where relevant, dim RGB elsewhere
    rgb_np = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
    mask = (relevancy_map > threshold).cpu().numpy()
    
    composite = rgb_np.copy()
    composite[mask] = cv2.addWeighted(rgb_np[mask], 0.3, heatmap[mask], 0.7, 0)
    composite[~mask] = (rgb_np[~mask] * 0.3).astype(np.uint8)
    
    return heatmap, composite


def find_localization_point(relevancy_map, kernel_size=30):
    """Find the point of maximum relevancy with smoothing."""
    np_relev = relevancy_map.cpu().numpy()
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    smoothed = cv2.filter2D(np_relev, -1, kernel)
    
    max_val = smoothed.max()
    max_loc = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    
    return max_loc[::-1], max_val  # Return (x, y), value


def create_visualization_figure(rgb_image, query, relevancy_maps, gt_mask=None, gt_bbox=None, loc_point=None):
    """Create a figure with multiple visualizations."""
    n_levels = len(relevancy_maps)
    fig, axes = plt.subplots(2, n_levels + 1, figsize=(4 * (n_levels + 2), 8))
    
    # Row 1: Original + Heatmaps for each level
    rgb_np = (rgb_image.cpu().numpy() * 255).astype(np.uint8)
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title(f'Query: "{query}"')
    axes[0, 0].axis('off')
    
    for i, rel_map in enumerate(relevancy_maps):
        heatmap = apply_colormap(rel_map)
        axes[0, i + 1].imshow(heatmap)
        axes[0, i + 1].set_title(f'Level {i + 1} Heatmap')
        axes[0, i + 1].axis('off')
    
    # Row 2: GT comparison + Composites
    if gt_mask is not None:
        gt_viz = rgb_np.copy()
        gt_viz[gt_mask > 0] = [0, 255, 0]  # Green overlay for GT
        axes[1, 0].imshow(cv2.addWeighted(rgb_np, 0.5, gt_viz, 0.5, 0))
        if gt_bbox is not None:
            for bbox in np.array(gt_bbox).reshape(-1, 4):
                x1, y1, x2, y2 = bbox.astype(int)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     fill=False, color='green', linewidth=2)
                axes[1, 0].add_patch(rect)
        axes[1, 0].set_title('Ground Truth')
    else:
        axes[1, 0].imshow(rgb_np)
        axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    
    for i, rel_map in enumerate(relevancy_maps):
        _, composite = visualize_relevancy(rel_map, rgb_image)
        axes[1, i + 1].imshow(composite)
        
        # Add localization point
        if loc_point is not None and i == loc_point[0]:  # Best level
            axes[1, i + 1].scatter([loc_point[1][0]], [loc_point[1][1]], 
                                   c='red', s=100, marker='x', linewidths=3)
        
        axes[1, i + 1].set_title(f'Level {i + 1} Composite')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="LangSplat Visualization")
    parser.add_argument("--dataset_name", type=str, default="figurines")
    parser.add_argument("--text_queries", nargs='+', type=str, default=None,
                        help="Text queries to search for objects")
    parser.add_argument("--use_gt_labels", action='store_true',
                        help="Use ground truth labels as queries")
    parser.add_argument("--feat_dir", type=str, default="output",
                        help="Directory containing rendered features")
    parser.add_argument("--model_path", "-m", type=str, default=None,
                        help="Base model path (same as -m in train/render). "
                             "The script appends _{1,2,3} for the three feature levels. "
                             "If provided, --feat_dir and --dataset_name are not used for feature paths.")
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt",
                        help="Autoencoder checkpoint directory")
    parser.add_argument("--gt_folder", type=str, default="lerf_ovs/label",
                        help="Ground truth labels folder")
    parser.add_argument("--output_dir", type=str, default="visualization_output",
                        help="Output directory for visualizations")
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument('--encoder_dims', nargs='+', type=int, 
                        default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[16, 32, 64, 128, 256, 256, 512])
    parser.add_argument("--frame_indices", nargs='+', type=int, default=None,
                        help="Specific frame indices to visualize (default: GT frames)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth
    gt_annotations, gt_labels = load_gt_annotations(args.gt_folder, args.dataset_name)
    print(f"Found GT labels: {gt_labels}")
    print(f"GT frame indices: {list(gt_annotations.keys())}")
    
    # Determine queries
    if args.use_gt_labels:
        queries = gt_labels
    elif args.text_queries:
        queries = args.text_queries
    else:
        queries = gt_labels  # Default to GT labels
    print(f"Using queries: {queries}")
    
    # Determine frames to visualize
    if args.frame_indices:
        eval_indices = args.frame_indices
    else:
        eval_indices = list(gt_annotations.keys())
    print(f"Visualizing frames: {eval_indices}")
    
    # Load feature paths for all 3 levels
    if args.model_path:
        # Use --model_path directly: append _{level} just like train/render do
        feat_dirs = [
            os.path.join(args.model_path + f"_{i}", "train/ours_None/renders_npy")
            for i in range(1, 4)
        ]
    else:
        # Legacy fallback: construct from --feat_dir and --dataset_name
        feat_dirs = [
            os.path.join(args.feat_dir, f"{args.dataset_name}_{i}", "train/ours_None/renders_npy")
            for i in range(1, 4)
        ]
    
    # Check if feature directories exist
    for i, fd in enumerate(feat_dirs):
        if not os.path.exists(fd):
            print(f"Warning: Feature directory not found: {fd}")
            print(f"Make sure you've run render.py for all levels (1, 2, 3)")
            return
    
    # Load features
    print("Loading rendered features...")
    feat_paths_all_levels = []
    for fd in feat_dirs:
        feat_paths = sorted(glob.glob(os.path.join(fd, '*.npy')),
                           key=lambda x: int(os.path.basename(x).split('.npy')[0]))
        feat_paths_all_levels.append(feat_paths)
    
    n_frames = len(feat_paths_all_levels[0])
    print(f"Found {n_frames} rendered frames")
    
    # Get image shape from first feature
    sample_feat = np.load(feat_paths_all_levels[0][0])
    h, w, _ = sample_feat.shape
    print(f"Feature shape: {h}x{w}")
    
    # Load autoencoder
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.dataset_name, "best_ckpt.pth")
    if not os.path.exists(ae_ckpt_path):
        print(f"Autoencoder checkpoint not found: {ae_ckpt_path}")
        return
    
    print("Loading autoencoder...")
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    autoencoder = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model = OpenCLIPNetwork(device)
    
    # Load RGB images
    rgb_dir = os.path.join("lerf_ovs", args.dataset_name, "images")
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    
    print(f"\nProcessing {len(eval_indices)} frames with {len(queries)} queries...")
    
    # Process each frame
    for idx in tqdm(eval_indices, desc="Processing frames"):
        if idx >= n_frames:
            print(f"Warning: Frame index {idx} out of range (max: {n_frames - 1})")
            continue
        
        # Load 3-dim features for all levels
        compressed_feats = []
        for level_paths in feat_paths_all_levels:
            feat = np.load(level_paths[idx])
            compressed_feats.append(feat)
        compressed_feats = np.stack(compressed_feats)  # 3 x H x W x 3
        compressed_feats = torch.from_numpy(compressed_feats).float().to(device)
        
        # Decode to 512-dim
        with torch.no_grad():
            n_levels, fh, fw, _ = compressed_feats.shape
            restored_feat = autoencoder.decode(compressed_feats.flatten(0, 2))
            restored_feat = restored_feat.view(n_levels, fh, fw, -1)  # 3 x H x W x 512
        
        # Load RGB image
        rgb_img = cv2.imread(rgb_paths[idx])[..., ::-1]  # BGR to RGB
        rgb_img = cv2.resize(rgb_img, (fw, fh))
        rgb_img = torch.from_numpy(rgb_img / 255.0).float().to(device)
        
        # Get GT annotation if available
        gt_ann = gt_annotations.get(idx, None)
        
        # Process each query
        for query in queries:
            clip_model.set_positives([query])
            
            # Get relevancy map for all levels
            relevancy_map = clip_model.get_max_across(restored_feat)  # n_levels x 1 x H x W
            relevancy_maps = [relevancy_map[i, 0] for i in range(n_levels)]
            
            # Find best level and localization point
            best_level = 0
            best_score = 0
            best_loc = None
            for i, rel_map in enumerate(relevancy_maps):
                loc, score = find_localization_point(rel_map)
                if score > best_score:
                    best_score = score
                    best_level = i
                    best_loc = loc
            
            # Get GT mask and bbox if available
            gt_mask = None
            gt_bbox = None
            if gt_ann is not None:
                for obj in gt_ann['objects']:
                    if obj['label'] == query:
                        gt_mask = cv2.resize(obj['mask'], (fw, fh))
                        gt_bbox = obj['bbox']
                        break
            
            # Create visualization
            fig = create_visualization_figure(
                rgb_img, query, relevancy_maps,
                gt_mask=gt_mask, gt_bbox=gt_bbox,
                loc_point=(best_level, best_loc)
            )
            
            # Save
            save_name = f"frame_{idx:05d}_{query.replace(' ', '_')}.png"
            save_path = output_path / save_name
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Also save individual heatmaps
            for i, rel_map in enumerate(relevancy_maps):
                heatmap = apply_colormap(rel_map)
                heatmap_path = output_path / f"frame_{idx:05d}_{query.replace(' ', '_')}_level{i+1}_heatmap.png"
                cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    
    print(f"\nVisualization complete! Results saved to: {output_path}")
    print(f"Generated {len(eval_indices) * len(queries)} visualization sets")


if __name__ == "__main__":
    main()
