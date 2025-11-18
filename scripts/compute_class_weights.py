"""Compute class weights for imbalanced segmentation training.

This script analyzes mask files to compute:
1. Per-class pixel counts (frequency)
2. Per-class inverse frequency weights (for CrossEntropyLoss)
3. Per-image class presence (how many images contain each class)

Usage:
    python scripts/compute_class_weights.py --mask_dir ./dataset/mask --output class_weights.json
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import img2seg, PALETTE

def compute_class_statistics(mask_dir: str, output_path: str):
    """Compute class statistics from mask directory."""
    mask_dir = Path(mask_dir)
    mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
    
    if not mask_files:
        print(f"Error: No mask files found in {mask_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    
    num_classes = len(PALETTE)
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    class_image_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Process each mask
    for mask_path in tqdm(mask_files, desc="Processing masks"):
        try:
            # Use img2seg to get class indices (sanitized to nearest palette color)
            seg = img2seg(str(mask_path))
            
            # Count pixels per class
            unique_classes, counts = np.unique(seg, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                if 0 <= cls < num_classes:
                    class_pixel_counts[cls] += count
                    
            # Count image presence (which classes appear in this image)
            for cls in unique_classes:
                if 0 <= cls < num_classes:
                    class_image_counts[cls] += 1
                    
        except Exception as e:
            print(f"Warning: Failed to process {mask_path}: {e}")
            continue
    
    # Compute statistics
    total_pixels = class_pixel_counts.sum()
    class_frequencies = class_pixel_counts / total_pixels if total_pixels > 0 else class_pixel_counts
    
    # Compute inverse frequency weights (with smoothing to avoid division by zero)
    # Formula: weight = 1 / (frequency + epsilon)
    epsilon = 1e-7
    inverse_freq_weights = 1.0 / (class_frequencies + epsilon)
    # Normalize weights so that mean weight = 1.0
    inverse_freq_weights = inverse_freq_weights / inverse_freq_weights.mean()
    
    # Alternative: median frequency balancing
    # median_freq = np.median(class_frequencies[class_frequencies > 0])
    # median_weights = median_freq / (class_frequencies + epsilon)
    
    # Build results
    class_names = ['background', 'hair', 'eye', 'mouth', 'face', 'skin', 'clothes', 'others']
    results = {
        'num_classes': num_classes,
        'num_images': len(mask_files),
        'total_pixels': int(total_pixels),
        'class_names': class_names[:num_classes],
        'class_statistics': []
    }
    
    print("\n=== Class Statistics ===")
    print(f"{'Class':<12} {'Pixels':>12} {'Frequency':>12} {'Images':>8} {'Weight':>10}")
    print("-" * 70)
    
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f'class_{i}'
        pixel_count = int(class_pixel_counts[i])
        frequency = float(class_frequencies[i])
        image_count = int(class_image_counts[i])
        weight = float(inverse_freq_weights[i])
        
        results['class_statistics'].append({
            'class_id': i,
            'class_name': class_name,
            'pixel_count': pixel_count,
            'frequency': frequency,
            'image_count': image_count,
            'images_percentage': float(image_count / len(mask_files) * 100),
            'inverse_freq_weight': weight,
        })
        
        print(f"{class_name:<12} {pixel_count:>12,} {frequency:>11.4%} {image_count:>8} {weight:>10.4f}")
    
    # Add weight tensor (as list for JSON serialization)
    results['weights_tensor'] = inverse_freq_weights.tolist()
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Statistics saved to {output_path}")
    print(f"\nRecommended class weights for CrossEntropyLoss:")
    print(f"  torch.tensor({inverse_freq_weights.tolist()}, dtype=torch.float32)")
    
    # Identify rare classes
    rare_threshold = 0.05  # classes appearing in < 5% of images
    rare_classes = []
    for stat in results['class_statistics']:
        if stat['images_percentage'] < rare_threshold * 100:
            rare_classes.append(stat['class_name'])
    
    if rare_classes:
        print(f"\n⚠ Warning: Rare classes detected (< 5% of images): {', '.join(rare_classes)}")
        print("  Consider: 1) Using class weights (already computed above)")
        print("           2) Oversampling images containing rare classes")
        print("           3) Verifying mask generation for these classes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute class weights from segmentation masks')
    parser.add_argument('--mask_dir', type=str, default='./dataset/mask',
                        help='Directory containing mask images')
    parser.add_argument('--output', type=str, default='class_weights.json',
                        help='Output JSON file for statistics')
    args = parser.parse_args()
    
    compute_class_statistics(args.mask_dir, args.output)
