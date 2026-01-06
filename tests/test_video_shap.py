"""
Tests for VideoSHAP module.

Tests data structures, manipulators, and SHAP calculation logic.
Uses mocks for expensive model calls.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import cv2

# Import VideoSHAP components
from token_shap.video_shap import (
    TrackedObject,
    ObjectFrame,
    VideoTrackingResult,
    VideoBlurManipulator,
    VideoBlackoutManipulator,
    VideoInpaintManipulator,
    VideoSHAP,
    VideoSHAPVisualizer,
    SAM3VideoSegmentationModel,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_frames():
    """Create sample video frames (5 frames, 100x100 RGB)"""
    frames = []
    for i in range(5):
        # Create frame with checkerboard pattern (high contrast for blur testing)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a checkerboard pattern that blur will definitely change
        for y in range(100):
            for x in range(100):
                # Checkerboard with 5px squares
                if ((x // 5) + (y // 5)) % 2 == 0:
                    frame[y, x] = [255, 0, 0]  # Red
                else:
                    frame[y, x] = [0, 255, 0]  # Green
        frames.append(frame)
    return frames


@pytest.fixture
def sample_mask():
    """Create a sample binary mask (100x100)"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Square object in center
    mask[30:70, 30:70] = 1
    return mask


@pytest.fixture
def sample_tracked_object(sample_mask):
    """Create a sample tracked object"""
    frames = {}
    for i in range(5):
        frames[i] = ObjectFrame(
            mask=sample_mask.copy(),
            bbox=(30, 30, 70, 70),
            score=0.9
        )
    return TrackedObject(
        object_id=0,
        label="person",
        frames=frames,
        confidence=0.9
    )


@pytest.fixture
def sample_tracking_result(sample_frames, sample_tracked_object):
    """Create a sample tracking result"""
    # Create second object
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[10:30, 60:90] = 1

    frames2 = {}
    for i in range(5):
        frames2[i] = ObjectFrame(
            mask=mask2.copy(),
            bbox=(60, 10, 90, 30),
            score=0.8
        )

    obj2 = TrackedObject(
        object_id=1,
        label="car",
        frames=frames2,
        confidence=0.8
    )

    return VideoTrackingResult(
        video_frames=sample_frames,
        objects={0: sample_tracked_object, 1: obj2},
        fps=10.0,
        resolution=(100, 100)
    )


# =============================================================================
# Test Data Classes
# =============================================================================

class TestObjectFrame:
    """Tests for ObjectFrame dataclass"""

    def test_creation(self, sample_mask):
        """Test ObjectFrame creation"""
        obj_frame = ObjectFrame(
            mask=sample_mask,
            bbox=(30, 30, 70, 70),
            score=0.95
        )

        assert obj_frame.mask.shape == (100, 100)
        assert obj_frame.bbox == (30, 30, 70, 70)
        assert obj_frame.score == 0.95


class TestTrackedObject:
    """Tests for TrackedObject dataclass"""

    def test_creation(self, sample_tracked_object):
        """Test TrackedObject creation"""
        obj = sample_tracked_object

        assert obj.object_id == 0
        assert obj.label == "person"
        assert obj.confidence == 0.9
        assert obj.num_frames == 5

    def test_frame_indices(self, sample_tracked_object):
        """Test frame_indices property"""
        obj = sample_tracked_object

        assert obj.frame_indices == [0, 1, 2, 3, 4]

    def test_num_frames(self, sample_tracked_object):
        """Test num_frames property"""
        obj = sample_tracked_object

        assert obj.num_frames == 5


class TestVideoTrackingResult:
    """Tests for VideoTrackingResult dataclass"""

    def test_creation(self, sample_tracking_result):
        """Test VideoTrackingResult creation"""
        result = sample_tracking_result

        assert result.num_frames == 5
        assert result.num_objects == 2
        assert result.fps == 10.0
        assert result.resolution == (100, 100)

    def test_get_object_labels(self, sample_tracking_result):
        """Test get_object_labels method"""
        result = sample_tracking_result
        labels = result.get_object_labels()

        assert "person" in labels
        assert "car" in labels
        assert len(labels) == 2


# =============================================================================
# Test Video Manipulators
# =============================================================================

class TestVideoBlurManipulator:
    """Tests for VideoBlurManipulator"""

    def test_mask_object(self, sample_frames, sample_tracked_object):
        """Test that blurring modifies frames correctly"""
        manipulator = VideoBlurManipulator(blur_radius=11)

        result = manipulator.mask_object(sample_frames, sample_tracked_object)

        # Should have same number of frames
        assert len(result) == len(sample_frames)

        # Frames should be modified in masked region
        # (blurred values differ from original)
        for i, (orig, mod) in enumerate(zip(sample_frames, result)):
            mask = sample_tracked_object.frames[i].mask

            # Check that masked region is different (blurred)
            masked_orig = orig[mask > 0]
            masked_mod = mod[mask > 0]

            # Values should be different due to blur
            # (unless original was uniform, which ours isn't)
            assert not np.array_equal(masked_orig, masked_mod)

    def test_blur_radius_odd(self):
        """Test that blur radius is made odd"""
        manipulator = VideoBlurManipulator(blur_radius=10)
        assert manipulator.blur_radius == 11


class TestVideoBlackoutManipulator:
    """Tests for VideoBlackoutManipulator"""

    def test_mask_object_black(self, sample_frames, sample_tracked_object):
        """Test blackout with default black color"""
        manipulator = VideoBlackoutManipulator()

        result = manipulator.mask_object(sample_frames, sample_tracked_object)

        # Check masked region is black
        for i, frame in enumerate(result):
            mask = sample_tracked_object.frames[i].mask
            masked_pixels = frame[mask > 0]

            assert np.all(masked_pixels == 0)

    def test_mask_object_custom_color(self, sample_frames, sample_tracked_object):
        """Test blackout with custom fill color"""
        manipulator = VideoBlackoutManipulator(fill_color=(255, 0, 0))  # Red

        result = manipulator.mask_object(sample_frames, sample_tracked_object)

        # Check masked region is red
        for i, frame in enumerate(result):
            mask = sample_tracked_object.frames[i].mask
            masked_pixels = frame[mask > 0]

            # All pixels should be (255, 0, 0)
            assert np.all(masked_pixels[:, 0] == 255)
            assert np.all(masked_pixels[:, 1] == 0)
            assert np.all(masked_pixels[:, 2] == 0)


class TestVideoInpaintManipulator:
    """Tests for VideoInpaintManipulator"""

    def test_mask_object(self, sample_frames, sample_tracked_object):
        """Test inpainting modifies frames"""
        manipulator = VideoInpaintManipulator(inpaint_radius=3)

        result = manipulator.mask_object(sample_frames, sample_tracked_object)

        # Should have same number of frames
        assert len(result) == len(sample_frames)

        # Frames should be modified
        for orig, mod in zip(sample_frames, result):
            # At least some pixels should be different
            assert not np.array_equal(orig, mod)


class TestVideoManipulatorMultiObject:
    """Tests for masking multiple objects"""

    def test_create_masked_video(self, sample_tracking_result):
        """Test masking multiple objects"""
        manipulator = VideoBlackoutManipulator()

        objects_to_mask = list(sample_tracking_result.objects.values())
        result = manipulator.create_masked_video(
            sample_tracking_result.video_frames,
            objects_to_mask
        )

        # Both object regions should be black
        for i, frame in enumerate(result):
            for obj in objects_to_mask:
                if i in obj.frames:
                    mask = obj.frames[i].mask
                    masked_pixels = frame[mask > 0]
                    assert np.all(masked_pixels == 0)


# =============================================================================
# Test VideoSHAPVisualizer
# =============================================================================

class TestVideoSHAPVisualizer:
    """Tests for VideoSHAPVisualizer"""

    def test_create_heatmap_video(self, sample_tracking_result):
        """Test heatmap video creation"""
        visualizer = VideoSHAPVisualizer()

        shapley_values = {
            "person_1": 0.7,
            "car_2": 0.3
        }

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name

        try:
            result_path = visualizer.create_heatmap_video(
                frames=sample_tracking_result.video_frames,
                objects=sample_tracking_result.objects,
                shapley_values=shapley_values,
                fps=10.0,
                output_path=output_path,
                show_colorbar=True,
                heatmap_opacity=0.5
            )

            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

            # Verify video can be read
            cap = cv2.VideoCapture(result_path)
            assert cap.isOpened()
            ret, frame = cap.read()
            assert ret
            cap.release()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_create_highlight_video(self, sample_tracking_result):
        """Test highlight video creation"""
        visualizer = VideoSHAPVisualizer()

        shapley_values = {
            "person_1": 0.7,
            "car_2": 0.3
        }

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name

        try:
            result_path = visualizer.create_highlight_video(
                frames=sample_tracking_result.video_frames,
                objects=sample_tracking_result.objects,
                shapley_values=shapley_values,
                fps=10.0,
                output_path=output_path,
                top_k=1,
                dim_factor=0.3
            )

            assert os.path.exists(result_path)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# Test VideoSHAP Class
# =============================================================================

class TestVideoSHAPSamples:
    """Test VideoSHAP._get_samples method"""

    def test_get_samples_from_tracking_result(self, sample_tracking_result):
        """Test that _get_samples returns tracked objects"""
        # Create mock model and vectorizer
        mock_model = Mock()
        mock_vectorizer = Mock()
        mock_seg_model = Mock()
        mock_manipulator = Mock()

        video_shap = VideoSHAP(
            model=mock_model,
            segmentation_model=mock_seg_model,
            manipulator=mock_manipulator,
            vectorizer=mock_vectorizer
        )

        samples = video_shap._get_samples(sample_tracking_result)

        assert len(samples) == 2
        assert all(isinstance(s, TrackedObject) for s in samples)


class TestVideoSHAPCombinationKey:
    """Test VideoSHAP._get_combination_key method"""

    def test_combination_key(self, sample_tracked_object):
        """Test combination key generation"""
        mock_model = Mock()
        mock_vectorizer = Mock()
        mock_seg_model = Mock()
        mock_manipulator = Mock()

        video_shap = VideoSHAP(
            model=mock_model,
            segmentation_model=mock_seg_model,
            manipulator=mock_manipulator,
            vectorizer=mock_vectorizer
        )

        key = video_shap._get_combination_key(
            [sample_tracked_object],
            (1,)
        )

        assert "person" in key
        assert "(1,)" in key


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================

class TestVideoSHAPIntegration:
    """Integration tests for VideoSHAP with mocked external services"""

    def test_analyze_with_mocks(self, sample_tracking_result):
        """Test full analysis pipeline with mocked components"""
        # Setup mocks
        mock_model = Mock()
        mock_model.generate_from_video.return_value = "The person is walking"

        # The vectorizer needs to return appropriate sized arrays
        # For 2 objects, we have 2 essential combinations + 1 baseline = 3 texts total
        mock_vectorizer = Mock()
        mock_vectorizer.vectorize.return_value = np.random.randn(3, 128)  # 3 texts (baseline + 2 combos), 128 dims
        mock_vectorizer.calculate_similarity.return_value = np.random.rand(2)  # 2 comparisons (for 2 combos)

        mock_seg_model = Mock()
        mock_seg_model.track.return_value = sample_tracking_result

        mock_manipulator = Mock()
        mock_manipulator.create_masked_video.return_value = sample_tracking_result.video_frames

        # Create VideoSHAP
        video_shap = VideoSHAP(
            model=mock_model,
            segmentation_model=mock_seg_model,
            manipulator=mock_manipulator,
            vectorizer=mock_vectorizer,
            debug=True
        )

        # Create temp video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_video = f.name
            # Write a minimal video
            import imageio
            imageio.mimwrite(temp_video, sample_tracking_result.video_frames, fps=10)

        try:
            # Run analysis
            results_df, shapley_values = video_shap.analyze(
                video_path=temp_video,
                prompt="What is happening?",
                text_prompts=["person", "car"],
                sampling_ratio=0.0,  # Only essential combinations
                max_combinations=10
            )

            # Verify results
            assert shapley_values is not None
            assert len(shapley_values) == 2  # Two objects

            # Verify mocks were called
            mock_seg_model.track.assert_called_once()
            assert mock_model.generate_from_video.call_count >= 1

        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)


# =============================================================================
# Test SAM3VideoSegmentationModel (unit tests with mocks)
# =============================================================================

class TestSAM3VideoSegmentationModel:
    """Unit tests for SAM3VideoSegmentationModel"""

    def test_initialization_lazy(self):
        """Test that model initialization is lazy"""
        model = SAM3VideoSegmentationModel(verbose=False)

        # Model should not be loaded yet
        assert model.sam_model is None
        assert model.sam_processor is None
        assert model._initialized is False

    def test_get_bbox_from_mask(self, sample_mask):
        """Test bounding box extraction from mask"""
        model = SAM3VideoSegmentationModel(verbose=False)

        bbox = model._get_bbox_from_mask(sample_mask)

        assert bbox is not None
        assert bbox == (30, 30, 69, 69)  # Based on sample_mask fixture

    def test_get_bbox_from_empty_mask(self):
        """Test bbox from empty mask returns None"""
        model = SAM3VideoSegmentationModel(verbose=False)

        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        bbox = model._get_bbox_from_mask(empty_mask)

        assert bbox is None


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
