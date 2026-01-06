#!/usr/bin/env python3
"""
Extract key frames from test videos for visual review.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def extract_frame(video_path, frame_num=10):
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def create_comparison_grid(output_dir):
    """Create a comparison grid of all videos."""
    output_dir = Path(output_dir)

    # Define groups
    groups = [
        ("Original vs Blur", [
            "01_original.mp4",
            "02_blur_obj0_red_rectangle.mp4",
            "02_blur_obj1_green_circle.mp4",
            "02_blur_obj2_blue_triangle.mp4",
        ]),
        ("Blackout Variations", [
            "01_original.mp4",
            "03_blackout_all_objects.mp4",
            "04_blackout_red.mp4",
            "04_blackout_gray.mp4",
        ]),
        ("Inpaint Results", [
            "01_original.mp4",
            "05_inpaint_obj0_red_rectangle.mp4",
            "05_inpaint_obj1_green_circle.mp4",
            "05_inpaint_all_objects.mp4",
        ]),
        ("SHAP Visualization", [
            "01_original.mp4",
            "06_heatmap_visualization.mp4",
            "07_highlight_top1.mp4",
            "07_highlight_top2.mp4",
        ]),
        ("Combination Masking", [
            "01_original.mp4",
            "08_combo_hide_obj0.mp4",
            "08_combo_hide_obj0_obj1.mp4",
            "08_combo_hide_obj1_obj2.mp4",
        ]),
    ]

    frame_num = 15  # Middle frame

    for group_name, videos in groups:
        print(f"\n{group_name}:")
        print("-" * 50)

        frames = []
        labels = []

        for video in videos:
            video_path = output_dir / video
            if video_path.exists():
                frame = extract_frame(video_path, frame_num)
                if frame is not None:
                    frames.append(frame)
                    labels.append(video.replace(".mp4", ""))
                    print(f"  ✓ {video}")
                else:
                    print(f"  ✗ {video} - could not extract frame")
            else:
                print(f"  ✗ {video} - file not found")

        if len(frames) >= 2:
            # Create comparison image
            h, w = frames[0].shape[:2]
            n_cols = min(4, len(frames))
            n_rows = (len(frames) + n_cols - 1) // n_cols

            grid = np.zeros((n_rows * (h + 40), n_cols * w, 3), dtype=np.uint8)

            for i, (frame, label) in enumerate(zip(frames, labels)):
                row = i // n_cols
                col = i % n_cols
                y_start = row * (h + 40) + 40
                x_start = col * w
                grid[y_start:y_start+h, x_start:x_start+w] = frame

                # Add label
                cv2.putText(grid, label[:30], (x_start + 5, row * (h + 40) + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save comparison image
            grid_path = output_dir / f"comparison_{group_name.lower().replace(' ', '_')}.png"
            cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {grid_path.name}")


def analyze_masking_quality(output_dir):
    """Analyze masking quality by comparing pixel values."""
    output_dir = Path(output_dir)

    print("\n" + "=" * 60)
    print("MASKING QUALITY ANALYSIS")
    print("=" * 60)

    frame_num = 15

    # Load original frame
    original = extract_frame(output_dir / "01_original.mp4", frame_num)
    if original is None:
        print("Could not load original frame")
        return

    # Define object regions (approximate based on frame 15)
    regions = {
        "red_rectangle": (125, 100, 205, 160),  # obj0: x=50+15*5=125
        "green_circle": (350, 90, 450, 190),     # obj1: center at (400, 140)
        "blue_triangle": (250, 350, 350, 430),   # obj2: stationary
    }

    # Compare with blurred versions
    print("\n1. BLUR MASKING ANALYSIS:")
    print("-" * 40)

    for obj_name, (x1, y1, x2, y2) in regions.items():
        blur_file = f"02_blur_obj{list(regions.keys()).index(obj_name)}_{obj_name}.mp4"
        blurred = extract_frame(output_dir / blur_file, frame_num)

        if blurred is not None:
            # Compare pixel values in the masked region
            orig_region = original[y1:y2, x1:x2]
            blur_region = blurred[y1:y2, x1:x2]

            diff = np.abs(orig_region.astype(float) - blur_region.astype(float))
            mean_diff = np.mean(diff)

            print(f"  {obj_name}:")
            print(f"    - Mean pixel difference in masked region: {mean_diff:.2f}")
            print(f"    - Blur applied: {'YES' if mean_diff > 10 else 'NO (possible issue)'}")

    # Compare with blackout versions
    print("\n2. BLACKOUT MASKING ANALYSIS:")
    print("-" * 40)

    for obj_name, (x1, y1, x2, y2) in regions.items():
        blackout_file = f"03_blackout_obj{list(regions.keys()).index(obj_name)}_{obj_name}.mp4"
        blackout = extract_frame(output_dir / blackout_file, frame_num)

        if blackout is not None:
            # Check if region is black (close to 0)
            black_region = blackout[y1:y2, x1:x2]
            mean_val = np.mean(black_region)

            print(f"  {obj_name}:")
            print(f"    - Mean pixel value in masked region: {mean_val:.2f}")
            print(f"    - Blackout applied: {'YES' if mean_val < 30 else 'NO (possible issue)'}")

    # Verify heatmap colors
    print("\n3. HEATMAP VISUALIZATION ANALYSIS:")
    print("-" * 40)

    heatmap = extract_frame(output_dir / "06_heatmap_visualization.mp4", frame_num)
    if heatmap is not None:
        # Check colors in each region
        # Red rectangle should be more red (high importance = 0.85)
        # Green circle should be more yellow/orange (medium = 0.45)
        # Blue triangle should be more green (low = 0.15)

        for obj_name, (x1, y1, x2, y2) in regions.items():
            region = heatmap[y1:y2, x1:x2]
            avg_color = np.mean(region, axis=(0, 1))

            print(f"  {obj_name}:")
            print(f"    - Avg RGB: ({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    output_dir = Path("/home/ubuntu/TokenSHAP/test_outputs")

    print("=" * 60)
    print("EXTRACTING FRAMES FOR VISUAL REVIEW")
    print("=" * 60)

    create_comparison_grid(output_dir)
    analyze_masking_quality(output_dir)

    print(f"\nComparison images saved to: {output_dir}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("comparison_*.png")):
        print(f"  - {f.name}")
