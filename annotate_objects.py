#!/usr/bin/env python
"""
LangSplat Object Annotation Script
Renders images with semantic text labels and bounding boxes overlaid on detected objects.

Usage:
    # Annotate specific objects
    python annotate_objects.py --dataset_name figurines --text_queries "robot" "waldo" "porcelain hand"
    
    # Use ground truth labels
    python annotate_objects.py --dataset_name figurines --use_gt_labels
    
    # Annotate all objects in single composite image
    python annotate_objects.py --dataset_name figurines --text_queries "robot" "waldo" --composite
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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))

from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork


# Color palette for different labels (BGR format for OpenCV, will convert for PIL)
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
    (0, 255, 128),    # Spring green
    (128, 255, 0),    # Lime
    (255, 0, 128),    # Pink
    (0, 128, 255),    # Sky blue
]


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
        
        frame_ann = {
            'frame_name': frame_name,
            'objects': []
        }
        
        for obj in gt_data['objects']:
            label = obj['category']
            all_labels.add(label)
            frame_ann['objects'].append({'label': label})
        
        annotations[idx] = frame_ann
    
    return annotations, list(all_labels)


def get_relevancy_mask(relevancy_map, threshold=0.5):
    """Convert relevancy map to binary mask."""
    rel_norm = relevancy_map - relevancy_map.min()
    rel_norm = rel_norm / (rel_norm.max() + 1e-9)
    mask = (rel_norm > threshold).cpu().numpy().astype(np.uint8)
    return mask, rel_norm.cpu().numpy()


def find_object_bbox_and_center(mask):
    """Find bounding box and center of object from mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 100:  # Min area threshold
        return None, None, None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = (x, y, x + w, y + h)
    
    # Get center using moments
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)
    else:
        center = (x + w // 2, y + h // 2)
    
    return bbox, center, largest_contour


def get_best_font(size=20):
    """Get the best available font."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    return ImageFont.load_default()


def draw_label_on_image(image, label, bbox, center, color, font, alpha=0.4):
    """Draw label with background box on image using PIL for better text rendering."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image, 'RGBA')
    
    x1, y1, x2, y2 = bbox
    
    # Draw semi-transparent bounding box
    overlay_color = (*color, int(255 * alpha))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw filled rectangle with transparency for object highlight
    # Create a separate overlay for the fill
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x1, y1, x2, y2], fill=(*color, int(255 * 0.2)))
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    draw = ImageDraw.Draw(pil_image)
    
    # Calculate text size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position label above bbox (or below if not enough space)
    padding = 5
    label_x = x1
    label_y = y1 - text_height - padding * 2
    
    if label_y < 0:
        label_y = y2 + padding
    
    # Draw label background
    label_bg = [
        label_x - padding,
        label_y - padding,
        label_x + text_width + padding,
        label_y + text_height + padding
    ]
    draw.rectangle(label_bg, fill=color)
    
    # Draw text (white on colored background)
    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
    
    # Convert back to numpy array
    return np.array(pil_image.convert('RGB'))


def draw_contour_on_image(image, contour, color, thickness=2):
    """Draw contour outline on image."""
    cv2.drawContours(image, [contour], -1, color, thickness)
    return image


def annotate_single_object(rgb_image, relevancy_map, label, color, font, threshold=0.5, draw_contour=True):
    """Annotate a single object on the image."""
    mask, rel_norm = get_relevancy_mask(relevancy_map, threshold)
    bbox, center, contour = find_object_bbox_and_center(mask)
    
    if bbox is None:
        return rgb_image, None
    
    # Draw contour
    annotated = rgb_image.copy()
    if draw_contour and contour is not None:
        annotated = draw_contour_on_image(annotated, contour, color, thickness=2)
    
    # Draw label
    annotated = draw_label_on_image(annotated, label, bbox, center, color, font)
    
    return annotated, {'label': label, 'bbox': bbox, 'center': center}


def create_composite_annotation(rgb_image, detections, font):
    """Create image with all detected objects annotated."""
    annotated = rgb_image.copy()
    
    for det in detections:
        label = det['label']
        color = det['color']
        contour = det.get('contour')
        bbox = det['bbox']
        center = det['center']
        
        # Draw contour
        if contour is not None:
            annotated = draw_contour_on_image(annotated, contour, color, thickness=2)
        
        # Draw label
        annotated = draw_label_on_image(annotated, label, bbox, center, color, font)
    
    return annotated


def process_frame(rgb_image, restored_feat, queries, clip_model, font, threshold=0.5, best_level_only=True):
    """Process a single frame and return annotated image with all queries."""
    n_levels = restored_feat.shape[0]
    detections = []
    
    for q_idx, query in enumerate(queries):
        color = COLORS[q_idx % len(COLORS)]
        
        clip_model.set_positives([query])
        relevancy_map = clip_model.get_max_across(restored_feat)  # n_levels x 1 x H x W
        
        # Use best level (usually level with highest max relevancy)
        if best_level_only:
            best_level = 0
            best_max = 0
            for i in range(n_levels):
                max_val = relevancy_map[i, 0].max().item()
                if max_val > best_max:
                    best_max = max_val
                    best_level = i
            rel_map = relevancy_map[best_level, 0]
        else:
            # Average across levels
            rel_map = relevancy_map[:, 0].mean(dim=0)
        
        mask, rel_norm = get_relevancy_mask(rel_map, threshold)
        bbox, center, contour = find_object_bbox_and_center(mask)
        
        if bbox is not None:
            detections.append({
                'label': query,
                'bbox': bbox,
                'center': center,
                'contour': contour,
                'color': color,
                'relevancy': rel_norm
            })
    
    # Create composite annotated image
    annotated = create_composite_annotation(rgb_image, detections, font)
    
    return annotated, detections


def main():
    parser = argparse.ArgumentParser(description="LangSplat Object Annotation")
    parser.add_argument("--dataset_name", type=str, default="figurines")
    parser.add_argument("--text_queries", nargs='+', type=str, default=None,
                        help="Text queries to search for objects")
    parser.add_argument("--use_gt_labels", action='store_true',
                        help="Use ground truth labels as queries")
    parser.add_argument("--feat_dir", type=str, default="output",
                        help="Directory containing rendered features")
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt",
                        help="Autoencoder checkpoint directory")
    parser.add_argument("--gt_folder", type=str, default="lerf_ovs/label",
                        help="Ground truth labels folder")
    parser.add_argument("--output_dir", type=str, default="annotated_output",
                        help="Output directory for annotated images")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Relevancy threshold for object detection")
    parser.add_argument('--encoder_dims', nargs='+', type=int, 
                        default=[256, 128, 64, 32, 3])
    parser.add_argument('--decoder_dims', nargs='+', type=int,
                        default=[16, 32, 64, 128, 256, 256, 512])
    parser.add_argument("--frame_indices", nargs='+', type=int, default=None,
                        help="Specific frame indices to annotate (default: all frames)")
    parser.add_argument("--font_size", type=int, default=18,
                        help="Font size for labels")
    parser.add_argument("--save_individual", action='store_true',
                        help="Save individual images per query in addition to composite")
    parser.add_argument("--rgb_dir", type=str, default=None,
                        help="Directory containing RGB images (default: lerf_ovs/<dataset>/images)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir) / args.dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth for labels if needed
    gt_annotations, gt_labels = {}, []
    if os.path.exists(args.gt_folder):
        try:
            gt_annotations, gt_labels = load_gt_annotations(args.gt_folder, args.dataset_name)
            print(f"Found GT labels: {gt_labels}")
        except Exception as e:
            print(f"Could not load GT annotations: {e}")
    
    # Determine queries
    if args.use_gt_labels and gt_labels:
        queries = gt_labels
    elif args.text_queries:
        queries = args.text_queries
    else:
        print("Error: Please provide --text_queries or use --use_gt_labels")
        return
    
    print(f"Using queries: {queries}")
    
    # Load feature paths for all 3 levels
    feat_dirs = [
        os.path.join(args.feat_dir, f"{args.dataset_name}_{i}", "train/ours_None/renders_npy")
        for i in range(1, 4)
    ]
    
    # Check if feature directories exist
    for i, fd in enumerate(feat_dirs):
        if not os.path.exists(fd):
            print(f"Error: Feature directory not found: {fd}")
            print(f"Make sure you've run render.py with --include_feature for all levels (1, 2, 3)")
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
    
    if n_frames == 0:
        print("Error: No rendered feature files found")
        return
    
    # Get image shape from first feature
    sample_feat = np.load(feat_paths_all_levels[0][0])
    h, w, _ = sample_feat.shape
    print(f"Feature shape: {h}x{w}")
    
    # Load autoencoder
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, args.dataset_name, "best_ckpt.pth")
    if not os.path.exists(ae_ckpt_path):
        print(f"Error: Autoencoder checkpoint not found: {ae_ckpt_path}")
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
    rgb_dir = args.rgb_dir if args.rgb_dir else os.path.join("lerf_ovs", args.dataset_name, "images")
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')) + 
                       glob.glob(os.path.join(rgb_dir, '*.png')))
    
    if not rgb_paths:
        print(f"Error: No RGB images found in {rgb_dir}")
        return
    
    print(f"Found {len(rgb_paths)} RGB images")
    
    # Determine frames to process
    if args.frame_indices:
        eval_indices = args.frame_indices
    else:
        eval_indices = list(range(min(n_frames, len(rgb_paths))))
    
    print(f"Processing {len(eval_indices)} frames...")
    
    # Get font
    font = get_best_font(args.font_size)
    
    # Process each frame
    for idx in tqdm(eval_indices, desc="Annotating frames"):
        if idx >= n_frames or idx >= len(rgb_paths):
            print(f"Warning: Frame index {idx} out of range")
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
        rgb_img = cv2.imread(rgb_paths[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (fw, fh))
        
        # Process frame with all queries
        annotated, detections = process_frame(
            rgb_img, restored_feat, queries, clip_model, font, 
            threshold=args.threshold
        )
        
        # Save composite annotated image
        save_path = output_path / f"frame_{idx:05d}_annotated.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        
        # Optionally save individual query results
        if args.save_individual:
            for det in detections:
                label = det['label']
                color = det['color']
                
                # Create individual annotation
                individual = rgb_img.copy()
                if det['contour'] is not None:
                    individual = draw_contour_on_image(individual, det['contour'], color, thickness=2)
                individual = draw_label_on_image(individual, label, det['bbox'], det['center'], color, font)
                
                ind_path = output_path / f"frame_{idx:05d}_{label.replace(' ', '_')}.png"
                cv2.imwrite(str(ind_path), cv2.cvtColor(individual, cv2.COLOR_RGB2BGR))
    
    print(f"\nAnnotation complete! Results saved to: {output_path}")
    print(f"Generated {len(eval_indices)} annotated images")


if __name__ == "__main__":
    main()
