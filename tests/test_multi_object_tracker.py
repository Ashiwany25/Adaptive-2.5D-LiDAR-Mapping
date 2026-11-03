import numpy as np
import pytest

from lidar_mapping.tracking.multi_object_tracker import (
    MultiObjectTracker,
)


def make_tracker():
    return MultiObjectTracker(
        max_matching_distance=3.0,
        max_missing_frames=2,
        frame_interval=0.1,
    )


def detection(x, y=0.0, z=0.0, semantic=10, instance=1):
    return {
        "centroid": np.array([x, y, z], dtype=float),
        "base_semantic_id": semantic,
        "instance_id": instance,
    }


def test_new_track_created():
    tracker = make_tracker()

    result = tracker.update([detection(0)])

    assert len(result) == 1
    assert result[0]["track_id"] == 1
    assert result[0]["association"] == "new"
    assert tracker.active_track_count() == 1


def test_same_object_keeps_track_id():
    tracker = make_tracker()

    tracker.update([detection(0)])
    result = tracker.update([detection(1)])

    assert result[0]["track_id"] == 1
    assert result[0]["association"] == "matched"


def test_velocity_estimation():
    tracker = make_tracker()

    tracker.update([detection(0)])
    result = tracker.update([detection(1)])

    np.testing.assert_allclose(
        result[0]["velocity"],
        [10.0, 0.0, 0.0],
    )

    assert result[0]["speed_mps"] == pytest.approx(10.0)


def test_motion_prediction_supports_matching():
    tracker = make_tracker()

    tracker.update([detection(0)])
    tracker.update([detection(2)])

    result = tracker.update([detection(4)])

    assert result[0]["track_id"] == 1
    assert result[0]["association"] == "matched"


def test_distant_object_creates_new_track():
    tracker = make_tracker()

    tracker.update([detection(0)])
    result = tracker.update([detection(10)])

    assert result[0]["track_id"] == 2
    assert result[0]["association"] == "new"
    assert tracker.active_track_count() == 2


def test_different_semantic_class_not_matched():
    tracker = make_tracker()

    tracker.update([detection(0, semantic=10)])

    result = tracker.update([
        detection(0.5, semantic=30)
    ])

    assert result[0]["track_id"] == 2
    assert tracker.active_track_count() == 2


def test_multiple_objects_keep_distinct_ids():
    tracker = make_tracker()

    tracker.update([
        detection(0, instance=101),
        detection(20, instance=102),
    ])

    result = tracker.update([
        detection(20.5, instance=102),
        detection(0.5, instance=101),
    ])

    ids = {
        item["instance_id"]: item["track_id"]
        for item in result
    }

    assert ids[101] == 1
    assert ids[102] == 2


def test_missing_track_remains_active():
    tracker = make_tracker()

    tracker.update([detection(0)])
    result = tracker.update([])

    assert result == []
    assert tracker.active_track_count() == 1
    assert tracker.tracks[1]["missing_frames"] == 1


def test_track_expires_after_missing_limit():
    tracker = make_tracker()

    tracker.update([detection(0)])

    tracker.update([])
    tracker.update([])

    assert tracker.active_track_count() == 1

    tracker.update([])

    assert tracker.active_track_count() == 0


def test_object_reappears_within_tolerance():
    tracker = make_tracker()

    tracker.update([detection(0)])
    tracker.update([])

    result = tracker.update([detection(1)])

    assert result[0]["track_id"] == 1
    assert result[0]["association"] == "matched"


def test_reset_clears_tracks():
    tracker = make_tracker()

    tracker.update([detection(0)])
    tracker.reset()

    assert tracker.active_track_count() == 0
    assert tracker.frame_index == -1

    result = tracker.update([detection(5)])

    assert result[0]["track_id"] == 1


@pytest.mark.parametrize(
    "centroid",
    [
        [np.nan, 0, 0],
        [np.inf, 0, 0],
        [1, 2],
    ],
)
def test_invalid_centroid_rejected(centroid):
    tracker = make_tracker()

    bad_detection = {
        "centroid": centroid,
        "base_semantic_id": 10,
        "instance_id": 1,
    }

    with pytest.raises(ValueError):
        tracker.update([bad_detection])
