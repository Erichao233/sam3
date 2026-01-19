#!/usr/bin/env python3
"""
Filter COCO-style annotation JSON to only include frames within the insertion phase.

This script reads insertion_end.txt and removes annotations for frames
beyond the insertion end point for each video sample.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Filter annotations to insertion-only")
    parser.add_argument("--input-json", type=str, required=True,
                        help="Input COCO annotation JSON file (e.g., train.json)")
    parser.add_argument("--output-json", type=str, required=True,
                        help="Output filtered annotation JSON file")
    parser.add_argument("--insertion-end-txt", type=str, required=True,
                        help="Path to insertion_end.txt with format: sample_id,end_frame,total_frames")
    parser.add_argument("--frames-root", type=str, default=None,
                        help="Optional: root directory for frames (to infer sample_id from image paths)")
    args = parser.parse_args()
    
    # Load insertion end info
    insertion_end = {}
    with open(args.insertion_end_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                sample_id = parts[0].strip()
                end_frame = int(parts[1].strip())
                insertion_end[sample_id] = end_frame
    
    print(f"Loaded insertion end info for {len(insertion_end)} samples")
    
    # Load original annotations
    with open(args.input_json) as f:
        coco = json.load(f)
    
    print(f"Original: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    # Build image_id -> image info mapping
    id_to_image = {img['id']: img for img in coco['images']}
    
    # Parse image file names to get sample_id and frame_idx
    # Expected format: sample_id/frame_idx.png or sample_id/frame_idx.jpg
    def parse_image_path(file_name):
        """Extract sample_id and frame_idx from image file name."""
        parts = Path(file_name).parts
        if len(parts) >= 2:
            sample_id = parts[-2]  # parent directory = sample_id
            frame_str = Path(parts[-1]).stem  # file name without extension
            try:
                frame_idx = int(frame_str)
                return sample_id, frame_idx
            except ValueError:
                pass
        return None, None
    
    # Filter images
    kept_image_ids = set()
    removed_images = 0
    
    for img in coco['images']:
        file_name = img.get('file_name', '')
        sample_id, frame_idx = parse_image_path(file_name)
        
        if sample_id is None or frame_idx is None:
            # Can't parse, keep the image
            kept_image_ids.add(img['id'])
            continue
        
        if sample_id not in insertion_end:
            # No insertion end info, keep all frames
            kept_image_ids.add(img['id'])
            continue
        
        end_frame = insertion_end[sample_id]
        if frame_idx <= end_frame:
            kept_image_ids.add(img['id'])
        else:
            removed_images += 1
    
    print(f"Kept {len(kept_image_ids)} images, removed {removed_images} images")
    
    # Filter images and annotations
    new_images = [img for img in coco['images'] if img['id'] in kept_image_ids]
    new_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in kept_image_ids]
    
    print(f"Filtered: {len(new_images)} images, {len(new_annotations)} annotations")
    
    # Create new COCO dict
    new_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco.get('categories', []),
    }
    
    # Copy any other top-level keys
    for key in coco:
        if key not in new_coco:
            new_coco[key] = coco[key]
    
    # Save
    with open(args.output_json, 'w') as f:
        json.dump(new_coco, f, indent=2)
    
    print(f"Saved filtered annotations to: {args.output_json}")
    
    # Summary stats per sample
    sample_stats = defaultdict(lambda: {'original': 0, 'kept': 0})
    for img in coco['images']:
        sample_id, _ = parse_image_path(img.get('file_name', ''))
        if sample_id:
            sample_stats[sample_id]['original'] += 1
    for img in new_images:
        sample_id, _ = parse_image_path(img.get('file_name', ''))
        if sample_id:
            sample_stats[sample_id]['kept'] += 1
    
    print("\n=== Per-sample summary (first 10) ===")
    for i, (sid, stats) in enumerate(sorted(sample_stats.items())[:10]):
        pct = 100 * stats['kept'] / max(1, stats['original'])
        print(f"  {sid}: {stats['original']} -> {stats['kept']} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
