import numpy as np
import pytest

from lidar_mapping.semantic.dynamic_separation import (
    separate_semantic_layers,
    summarize_semantic_layers,
)


def sample_data():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])

    semantic = np.array(
        [40, 10, 252, 99], dtype=np.int32
    )

    instance = np.array(
        [0, 101, 102, 0], dtype=np.int32
    )

    return points, semantic, instance


def test_static_environment_classification():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    assert result["static_mask"].tolist() == [
        True, False, False, False
    ]
    assert len(result["static_points"]) == 1


def test_potentially_movable_classification():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    assert result["movable_mask"].tolist() == [
        False, True, False, False
    ]


def test_annotated_moving_classification():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    assert result["moving_mask"].tolist() == [
        False, False, True, False
    ]


def test_object_layer_includes_both_categories():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    assert result["object_mask"].tolist() == [
        False, True, True, False
    ]

    assert len(result["object_points"]) == 2


def test_unknown_points_remain_separate():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    assert result["unknown_mask"].tolist() == [
        False, False, False, True
    ]


def test_instance_ids_preserved():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    np.testing.assert_array_equal(
        result["object_instance"],
        [101, 102],
    )


def test_summary_counts():
    points, semantic, instance = sample_data()

    result = separate_semantic_layers(
        points, semantic, instance
    )

    summary = summarize_semantic_layers(result)

    assert summary["total_points"] == 4
    assert summary["static_points"] == 1
    assert summary["potentially_movable_points"] == 1
    assert summary["annotated_moving_points"] == 1
    assert summary["unknown_points"] == 1
    assert summary["object_layer_points"] == 2


def test_empty_point_cloud():
    points = np.empty((0, 3))
    semantic = np.array([], dtype=np.int32)

    result = separate_semantic_layers(
        points, semantic
    )

    assert len(result["static_points"]) == 0
    assert len(result["object_points"]) == 0


def test_mismatched_label_count_rejected():
    points = np.zeros((2, 3))
    semantic = np.array([40], dtype=np.int32)

    with pytest.raises(ValueError):
        separate_semantic_layers(points, semantic)


def test_nonfinite_coordinates_rejected():
    points = np.array([
        [np.nan, 0.0, 0.0]
    ])

    semantic = np.array([40], dtype=np.int32)

    with pytest.raises(ValueError):
        separate_semantic_layers(points, semantic)
