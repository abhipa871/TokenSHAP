#!/usr/bin/env python3
"""
Visual verification test for VideoSHAP masking functionality.

This script creates test videos with tracked objects and applies different
masking strategies to verify they work correctly.
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_shap.video_shap import (
    TrackedObject,
    ObjectFrame,
    VideoTrackingResult,
    VideoBlurManipulator,
    VideoBlackoutManipulator,
    VideoInpaintManipulator,
    VideoSHAPVisualizer,
)
import imageio


def create_test_video_frames(num_frames=30, width=640, height=480):
    """Create test video frames with moving colored rectangles."""
    frames = []

    for i in range(num_frames):
        # Create a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient (blue to green)
        for y in range(height):
            ratio = y / height
            frame[y, :, 0] = int(50 + 50 * ratio)   # Blue
            frame[y, :, 1] = int(100 + 100 * ratio)  # Green
            frame[y, :, 2] = int(50 + 20 * ratio)   # Red

        # Add a checkerboard pattern for texture
        checker_size = 20
        for y in range(0, height, checker_size * 2):
            for x in range(0, width, checker_size * 2):
                frame[y:y+checker_size, x:x+checker_size] += 20
                frame[y+checker_size:y+checker_size*2, x+checker_size:x+checker_size*2] += 20

        # Draw a red rectangle (Object 1 - moving right)
        obj1_x = 50 + i * 5
        obj1_y = 100
        obj1_w, obj1_h = 80, 60
        cv2.rectangle(frame, (obj1_x, obj1_y), (obj1_x + obj1_w, obj1_y + obj1_h), (255, 50, 50), -1)
        cv2.putText(frame, "OBJ1", (obj1_x + 10, obj1_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a green circle (Object 2 - moving down)
        obj2_x = 400
        obj2_y = 80 + i * 4
        obj2_r = 50
        cv2.circle(frame, (obj2_x, obj2_y), obj2_r, (50, 255, 50), -1)
        cv2.putText(frame, "OBJ2", (obj2_x - 25, obj2_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Draw a blue triangle (Object 3 - stationary)
        obj3_pts = np.array([
            [300, 350],
            [250, 430],
            [350, 430]
        ], np.int32)
        cv2.fillPoly(frame, [obj3_pts], (50, 50, 255))
        cv2.putText(frame, "OBJ3", (275, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add frame number
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        frames.append(frame)

    return frames


def create_masks_for_objects(frames, num_frames=30, width=640, height=480):
    """Create masks for each tracked object."""
    objects = {}

    # Object 1: Red rectangle (moving right)
    obj1_frames = {}
    for i in range(num_frames):
        obj1_x = 50 + i * 5
        obj1_y = 100
        obj1_w, obj1_h = 80, 60

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[obj1_y:obj1_y+obj1_h, obj1_x:obj1_x+obj1_w] = 1

        obj1_frames[i] = ObjectFrame(
            mask=mask,
            bbox=(obj1_x, obj1_y, obj1_x + obj1_w, obj1_y + obj1_h),
            score=0.95
        )

    objects[0] = TrackedObject(
        object_id=0,
        label="red_rectangle",
        frames=obj1_frames,
        confidence=0.95
    )

    # Object 2: Green circle (moving down)
    obj2_frames = {}
    for i in range(num_frames):
        obj2_x = 400
        obj2_y = 80 + i * 4
        obj2_r = 50

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (obj2_x, obj2_y), obj2_r, 1, -1)

        obj2_frames[i] = ObjectFrame(
            mask=mask,
            bbox=(obj2_x - obj2_r, obj2_y - obj2_r, obj2_x + obj2_r, obj2_y + obj2_r),
            score=0.92
        )

    objects[1] = TrackedObject(
        object_id=1,
        label="green_circle",
        frames=obj2_frames,
        confidence=0.92
    )

    # Object 3: Blue triangle (stationary)
    obj3_frames = {}
    for i in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        obj3_pts = np.array([
            [300, 350],
            [250, 430],
            [350, 430]
        ], np.int32)
        cv2.fillPoly(mask, [obj3_pts], 1)

        obj3_frames[i] = ObjectFrame(
            mask=mask,
            bbox=(250, 350, 350, 430),
            score=0.88
        )

    objects[2] = TrackedObject(
        object_id=2,
        label="blue_triangle",
        frames=obj3_frames,
        confidence=0.88
    )

    return objects


def save_video(frames, output_path, fps=10):
    """Save frames as video."""
    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264',
        output_params=['-pix_fmt', 'yuv420p']
    )
    print(f"Saved: {output_path}")


def main():
    # Create output directory
    output_dir = Path("/home/ubuntu/TokenSHAP/test_outputs")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("VideoSHAP Visual Masking Verification Test")
    print("=" * 60)

    # Parameters
    num_frames = 30
    width, height = 640, 480
    fps = 10

    print(f"\nCreating test video: {num_frames} frames, {width}x{height}, {fps} FPS")

    # Create test frames
    frames = create_test_video_frames(num_frames, width, height)
    print(f"✓ Created {len(frames)} test frames")

    # Create tracked objects with masks
    objects = create_masks_for_objects(frames, num_frames, width, height)
    print(f"✓ Created {len(objects)} tracked objects with masks")

    for obj_id, obj in objects.items():
        print(f"  - Object {obj_id}: {obj.label} (confidence: {obj.confidence:.2f}, frames: {obj.num_frames})")

    # Create tracking result
    tracking_result = VideoTrackingResult(
        video_frames=frames,
        objects=objects,
        fps=fps,
        resolution=(width, height)
    )

    # Save original video
    print("\n" + "-" * 40)
    print("Saving original video...")
    save_video(frames, str(output_dir / "01_original.mp4"), fps)

    # Test 1: Blur Manipulator - mask each object individually
    print("\n" + "-" * 40)
    print("Testing VideoBlurManipulator...")
    blur_manipulator = VideoBlurManipulator(blur_radius=51)

    for obj_id, obj in objects.items():
        blurred_frames = blur_manipulator.mask_object([f.copy() for f in frames], obj)
        save_video(blurred_frames, str(output_dir / f"02_blur_obj{obj_id}_{obj.label}.mp4"), fps)

    # Blur all objects
    all_blurred = blur_manipulator.create_masked_video(frames, list(objects.values()))
    save_video(all_blurred, str(output_dir / "02_blur_all_objects.mp4"), fps)

    # Test 2: Blackout Manipulator
    print("\n" + "-" * 40)
    print("Testing VideoBlackoutManipulator...")
    blackout_manipulator = VideoBlackoutManipulator(fill_color=(0, 0, 0))

    for obj_id, obj in objects.items():
        blackout_frames = blackout_manipulator.mask_object([f.copy() for f in frames], obj)
        save_video(blackout_frames, str(output_dir / f"03_blackout_obj{obj_id}_{obj.label}.mp4"), fps)

    # Blackout all objects
    all_blackout = blackout_manipulator.create_masked_video(frames, list(objects.values()))
    save_video(all_blackout, str(output_dir / "03_blackout_all_objects.mp4"), fps)

    # Test 3: Colored Blackout
    print("\n" + "-" * 40)
    print("Testing VideoBlackoutManipulator with custom colors...")

    # Red blackout
    red_blackout = VideoBlackoutManipulator(fill_color=(255, 0, 0))
    red_frames = red_blackout.create_masked_video(frames, list(objects.values()))
    save_video(red_frames, str(output_dir / "04_blackout_red.mp4"), fps)

    # Gray blackout
    gray_blackout = VideoBlackoutManipulator(fill_color=(128, 128, 128))
    gray_frames = gray_blackout.create_masked_video(frames, list(objects.values()))
    save_video(gray_frames, str(output_dir / "04_blackout_gray.mp4"), fps)

    # Test 4: Inpaint Manipulator
    print("\n" + "-" * 40)
    print("Testing VideoInpaintManipulator...")
    inpaint_manipulator = VideoInpaintManipulator(inpaint_radius=5)

    for obj_id, obj in objects.items():
        inpainted_frames = inpaint_manipulator.mask_object([f.copy() for f in frames], obj)
        save_video(inpainted_frames, str(output_dir / f"05_inpaint_obj{obj_id}_{obj.label}.mp4"), fps)

    # Inpaint all objects
    all_inpainted = inpaint_manipulator.create_masked_video(frames, list(objects.values()))
    save_video(all_inpainted, str(output_dir / "05_inpaint_all_objects.mp4"), fps)

    # Test 5: VideoSHAP Visualizer - Heatmap
    print("\n" + "-" * 40)
    print("Testing VideoSHAPVisualizer heatmap...")
    visualizer = VideoSHAPVisualizer()

    # Create mock Shapley values
    shapley_values = {
        "red_rectangle_1": 0.85,  # High importance
        "green_circle_2": 0.45,   # Medium importance
        "blue_triangle_3": 0.15,  # Low importance
    }

    visualizer.create_heatmap_video(
        frames=frames,
        objects=objects,
        shapley_values=shapley_values,
        fps=fps,
        output_path=str(output_dir / "06_heatmap_visualization.mp4"),
        show_colorbar=True,
        heatmap_opacity=0.5,
        prompt="Which object is most important?"
    )
    print(f"Saved: {output_dir / '06_heatmap_visualization.mp4'}")

    # Test 6: VideoSHAP Visualizer - Highlight
    print("\n" + "-" * 40)
    print("Testing VideoSHAPVisualizer highlight...")

    visualizer.create_highlight_video(
        frames=frames,
        objects=objects,
        shapley_values=shapley_values,
        fps=fps,
        output_path=str(output_dir / "07_highlight_top1.mp4"),
        top_k=1,
        dim_factor=0.3
    )
    print(f"Saved: {output_dir / '07_highlight_top1.mp4'}")

    visualizer.create_highlight_video(
        frames=frames,
        objects=objects,
        shapley_values=shapley_values,
        fps=fps,
        output_path=str(output_dir / "07_highlight_top2.mp4"),
        top_k=2,
        dim_factor=0.3
    )
    print(f"Saved: {output_dir / '07_highlight_top2.mp4'}")

    # Test 7: Combination masking (mask specific combinations)
    print("\n" + "-" * 40)
    print("Testing combination masking (for SHAP analysis)...")

    # Mask only object 0 (show objects 1 and 2)
    combo_frames = blur_manipulator.create_masked_video(frames, [objects[0]])
    save_video(combo_frames, str(output_dir / "08_combo_hide_obj0.mp4"), fps)

    # Mask objects 0 and 1 (show only object 2)
    combo_frames = blur_manipulator.create_masked_video(frames, [objects[0], objects[1]])
    save_video(combo_frames, str(output_dir / "08_combo_hide_obj0_obj1.mp4"), fps)

    # Mask objects 1 and 2 (show only object 0)
    combo_frames = blur_manipulator.create_masked_video(frames, [objects[1], objects[2]])
    save_video(combo_frames, str(output_dir / "08_combo_hide_obj1_obj2.mp4"), fps)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.mp4")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    print("\n" + "-" * 40)
    print("VERIFICATION CHECKLIST:")
    print("-" * 40)
    print("[ ] 01_original.mp4 - Shows 3 distinct objects moving")
    print("[ ] 02_blur_*.mp4 - Each object should be blurred individually")
    print("[ ] 03_blackout_*.mp4 - Each object should be black")
    print("[ ] 04_blackout_*.mp4 - Objects filled with custom colors")
    print("[ ] 05_inpaint_*.mp4 - Objects should be removed/filled")
    print("[ ] 06_heatmap_*.mp4 - Objects colored by importance (red=high)")
    print("[ ] 07_highlight_*.mp4 - Top objects highlighted, others dimmed")
    print("[ ] 08_combo_*.mp4 - Specific object combinations masked")
    print("-" * 40)

    return output_dir


if __name__ == "__main__":
    output_dir = main()
