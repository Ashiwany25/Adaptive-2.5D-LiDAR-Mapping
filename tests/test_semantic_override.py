import numpy as np
import pytest

from lidar_mapping.semantic.resolution_override import (
    apply_semantic_override,
    summarize_semantic_override,
)


CONFIG = {
    "mapping": {
        "near_distance": 20.0,
        "mid_distance": 40.0,
        "near_resolution": 0.25,
        "mid_resolution": 0.5,
        "far_resolution": 1.0,
        "minimum_resolution": 0.25,
    },
    "semantic": {
        "confidence_threshold": 0.80,
        "critical_classes": [10, 11, 13, 15, 18, 20],
    },
}


def evaluate(distance, semantic_id, confidence=None):
    points = np.array([[distance, 0.0, 0.0]])
    labels = np.array([semantic_id], dtype=np.int32)

    scores = (
        None
        if confidence is None
        else np.array([confidence])
    )

    return apply_semantic_override(
        points=points,
        semantic_labels=labels,
        confidence=scores,
        sensor_position=np.zeros(3),
        config=CONFIG,
    )


def test_critical_object_refined_at_long_distance():
    result = evaluate(50.0, 10, 0.95)

    assert result["base_resolution"][0] == 1.0
    assert result["final_resolution"][0] == 0.25
    assert bool(result["override"][0])


def test_noncritical_object_keeps_base_resolution():
    result = evaluate(50.0, 40, 0.99)

    assert result["final_resolution"][0] == 1.0
    assert not bool(result["override"][0])


def test_low_confidence_blocks_refinement():
    result = evaluate(50.0, 10, 0.79)

    assert result["final_resolution"][0] == 1.0
    assert not bool(result["override_eligible"][0])
    assert not bool(result["override"][0])


def test_threshold_is_inclusive():
    result = evaluate(50.0, 10, 0.80)

    assert result["final_resolution"][0] == 0.25
    assert bool(result["override_eligible"][0])


def test_ground_truth_without_confidence():
    result = evaluate(50.0, 10)

    assert result["final_resolution"][0] == 0.25
    assert bool(result["confident"][0])


def test_already_fine_is_not_new_override():
    result = evaluate(10.0, 10, 0.95)

    assert result["base_resolution"][0] == 0.25
    assert result["final_resolution"][0] == 0.25
    assert not bool(result["override"][0])

    summary = summarize_semantic_override(result)

    assert summary["actual_refined_points"] == 0
    assert summary["fine_points_before"] == 1
    assert summary["fine_points_after"] == 1


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, np.nan, np.inf],
)
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        evaluate(50.0, 10, confidence)
