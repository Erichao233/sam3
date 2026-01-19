#!/usr/bin/env python3
"""
Convert EndoVis 2017 val sequences to videos with colored annotation overlays.
Each class gets a different transparent color overlay.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Color palette for 7 instrument classes (BGR for OpenCV, light/transparent)
# Using distinct colors with alpha for overlay
CLASS_COLORS = {
    1: (255, 150, 150),   # Light red - Bipolar Forceps
    2: (150, 255, 150),   # Light green - Prograsp Forceps
    3: (150, 150, 255),   # Light blue - Large Needle Driver
    4: (255, 255, 150),   # Light yellow - Vessel Sealer
    5: (255, 150, 255),   # Light magenta - Grasping Retractor
    6: (150, 255, 255),   # Light cyan - Monopolar Curved Scissors
    7: (200, 200, 200),   # Light gray - Other
}

CLASS_NAMES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def load_bmp_mask(path: Path) -> np.ndarray:
    """Load BMP mask where pixel value = class ID."""
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create colored overlay for each class in the mask."""
    overlay = image.copy()
    
    for class_id, color in CLASS_COLORS.items():
        class_mask = (mask == class_id)
        if class_mask.any():
            # Apply semi-transparent color
            overlay[class_mask] = (
                np.array(color) * alpha + 
                image[class_mask].astype(float) * (1 - alpha)
            ).astype(np.uint8)
    
    return overlay


def process_sequence(data_root: Path, seq_name: str, output_dir: Path, fps: int = 10):
    """Process one sequence and create video."""
    seq_dir = data_root / seq_name
    image_dir = seq_dir / "image"
    label_dir = seq_dir / "label"
    
    if not image_dir.exists() or not label_dir.exists():
        print(f"[skip] {seq_name}: missing image or label directory")
        return
    
    # Get sorted frames
    label_files = sorted(label_dir.glob("*.bmp"))
    if not label_files:
        print(f"[skip] {seq_name}: no label files found")
        return
    
    # Read first image to get dimensions
    first_image_path = None
    for ext in [".bmp", ".jpg", ".png"]:
        candidate = image_dir / f"{label_files[0].stem}{ext}"
        if candidate.exists():
            first_image_path = candidate
            break
    
    if first_image_path is None:
        print(f"[skip] {seq_name}: no matching image found")
        return
    
    with Image.open(first_image_path) as im:
        width, height = im.size
    
    # Create video writer
    output_path = output_dir / f"{seq_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Processing {seq_name}: {len(label_files)} frames -> {output_path}")
    
    for label_path in label_files:
        # Find matching image
        image_path = None
        for ext in [".bmp", ".jpg", ".png"]:
            candidate = image_dir / f"{label_path.stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            continue
        
        # Load image and mask
        with Image.open(image_path) as im:
            image = np.array(im.convert("RGB"))
        mask = load_bmp_mask(label_path)
        
        # Create overlay
        overlay = create_overlay(image, mask, alpha=0.35)
        
        # Add legend in corner
        legend_y = 30
        for class_id in sorted(CLASS_COLORS.keys()):
            if (mask == class_id).any():
                color = CLASS_COLORS[class_id]
                name = CLASS_NAMES[class_id]
                # Draw colored rectangle and text
                cv2.rectangle(overlay, (10, legend_y - 15), (25, legend_y), color, -1)
                cv2.putText(overlay, name, (30, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                legend_y += 20
        
        # Convert RGB to BGR for OpenCV
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        video_writer.write(overlay_bgr)
    
    video_writer.release()
    print(f"  Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create EndoVis annotation overlay videos")
    parser.add_argument("--data-root", type=str, default="/Users/eric/Desktop/SAM/endovis2017",
                        help="Path to endovis2017 directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_root/video)")
    parser.add_argument("--sequences", type=str, nargs="*", 
                        default=["val1", "val2", "val3", "val4", "val5", "val6", "val7", "val8", "val10"],
                        help="Sequences to process (default: val1-val8, val10, excluding val9)")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--alpha", type=float, default=0.35, help="Overlay transparency (0-1)")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Sequences: {args.sequences}")
    print()
    
    for seq_name in args.sequences:
        process_sequence(data_root, seq_name, output_dir, fps=args.fps)
    
    print(f"\nDone! Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
