#!/usr/bin/env python3
"""
Detect the end of needle insertion phase by analyzing GT mask area changes.

The insertion phase ends when the GT mask area starts to consistently decrease,
indicating the needle is being withdrawn and leaving behind tissue artifacts.

Output: insertion_end.txt with format: sample_id,end_frame
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict


def load_gt_areas(gt_dir: Path) -> list[tuple[int, int]]:
    """Load all GT masks and return (frame_idx, area) pairs sorted by frame."""
    areas = []
    for mask_path in gt_dir.glob("*.png"):
        try:
            frame_idx = int(mask_path.stem)
        except ValueError:
            continue
        mask = np.array(Image.open(mask_path))
        # Count non-zero pixels as mask area
        area = int((mask > 0).sum())
        areas.append((frame_idx, area))
    areas.sort(key=lambda x: x[0])
    return areas


def find_insertion_end(areas: list[tuple[int, int]], 
                       window_size: int = 10,
                       decrease_ratio: float = 0.15,
                       min_frames: int = 20) -> int | None:
    """
    Find the frame where insertion ends (needle starts being withdrawn).
    
    Strategy:
    1. Compute smoothed area curve (moving average)
    2. Find the peak area (maximum insertion depth)
    3. Return the frame at or just after the peak
    
    Args:
        areas: list of (frame_idx, area) tuples
        window_size: window for moving average smoothing
        decrease_ratio: minimum ratio of area decrease from peak to confirm withdrawal
        min_frames: minimum frames before considering peak detection
    
    Returns:
        Frame index where insertion ends, or None if can't detect
    """
    if len(areas) < min_frames:
        return None
    
    frame_indices = [a[0] for a in areas]
    area_values = np.array([a[1] for a in areas], dtype=float)
    
    # Handle empty masks
    if area_values.max() == 0:
        return None
    
    # Smooth the area curve
    kernel = np.ones(window_size) / window_size
    if len(area_values) >= window_size:
        smoothed = np.convolve(area_values, kernel, mode='same')
    else:
        smoothed = area_values
    
    # Find the global peak (max area = deepest insertion)
    peak_idx = np.argmax(smoothed)
    peak_area = smoothed[peak_idx]
    
    # Look for sustained decrease after peak
    # The insertion ends when area drops by `decrease_ratio` from peak
    threshold = peak_area * (1 - decrease_ratio)
    
    end_idx = peak_idx
    for i in range(peak_idx, len(smoothed)):
        if smoothed[i] < threshold:
            end_idx = i
            break
    else:
        # No significant decrease found, use peak as end
        end_idx = peak_idx
    
    # Return the frame index
    return frame_indices[end_idx]


def main():
    parser = argparse.ArgumentParser(description="Detect insertion end frame from GT masks")
    parser.add_argument("--gt-root", type=str, required=True,
                        help="Root directory containing GT masks (e.g., pork_dataset/gt)")
    parser.add_argument("--sample-txt", type=str, default=None,
                        help="Optional: file with sample IDs to process")
    parser.add_argument("--output", type=str, default="insertion_end.txt",
                        help="Output file path")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Smoothing window size")
    parser.add_argument("--decrease-ratio", type=float, default=0.15,
                        help="Area decrease ratio to detect withdrawal")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    args = parser.parse_args()
    
    gt_root = Path(args.gt_root)
    
    # Get sample directories
    if args.sample_txt:
        with open(args.sample_txt) as f:
            sample_ids = [line.strip() for line in f if line.strip()]
        sample_dirs = [gt_root / sid for sid in sample_ids]
    else:
        sample_dirs = sorted([d for d in gt_root.iterdir() if d.is_dir()])
    
    results = []
    
    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            print(f"[warn] Sample dir not found: {sample_dir}")
            continue
        
        sample_id = sample_dir.name
        areas = load_gt_areas(sample_dir)
        
        if not areas:
            print(f"[warn] No GT masks found for sample {sample_id}")
            continue
        
        end_frame = find_insertion_end(
            areas, 
            window_size=args.window_size,
            decrease_ratio=args.decrease_ratio
        )
        
        total_frames = max(a[0] for a in areas) + 1
        
        if end_frame is not None:
            results.append((sample_id, end_frame, total_frames))
            print(f"Sample {sample_id}: insertion ends at frame {end_frame} / {total_frames} "
                  f"({100*end_frame/total_frames:.1f}%)")
        else:
            # Use all frames if can't detect
            results.append((sample_id, total_frames - 1, total_frames))
            print(f"Sample {sample_id}: could not detect insertion end, using all {total_frames} frames")
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write("# sample_id,insertion_end_frame,total_frames\n")
        for sample_id, end_frame, total in results:
            f.write(f"{sample_id},{end_frame},{total}\n")
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} samples")
    print(f"Output saved to: {output_path}")
    
    # Optionally generate visualization
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results)))
            if len(results) == 1:
                axes = [axes]
            
            for ax, (sample_id, end_frame, total) in zip(axes, results):
                sample_dir = gt_root / sample_id
                areas = load_gt_areas(sample_dir)
                frames = [a[0] for a in areas]
                area_vals = [a[1] for a in areas]
                
                ax.plot(frames, area_vals, 'b-', label='GT area')
                ax.axvline(x=end_frame, color='r', linestyle='--', label=f'Insertion end: {end_frame}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Mask area (pixels)')
                ax.set_title(f'Sample {sample_id}')
                ax.legend()
            
            plt.tight_layout()
            viz_path = output_path.with_suffix('.png')
            plt.savefig(viz_path, dpi=100)
            print(f"Visualization saved to: {viz_path}")
        except ImportError:
            print("[warn] matplotlib not available, skipping visualization")


if __name__ == "__main__":
    main()
