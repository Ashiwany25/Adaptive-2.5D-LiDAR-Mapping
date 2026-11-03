import numpy as np
import pytest

from lidar_mapping.temporal.occupancy_persistence import (
    OccupancyPersistence,
    create_observation_grid,
)


def test_confirmation_after_three_frames():
    tracker = OccupancyPersistence((2, 2), persistence_frames=3)
    observed = np.array([[True, False], [False, False]])

    assert not tracker.update(observed)[0, 0]
    assert tracker.get_pending()[0, 0]

    assert not tracker.update(observed)[0, 0]
    assert tracker.update(observed)[0, 0]

    assert tracker.get_counters()[0, 0] == 3


def test_missing_observation_resets_streak():
    tracker = OccupancyPersistence((1, 1), persistence_frames=3)

    tracker.update(np.array([[True]]))
    tracker.update(np.array([[True]]))
    tracker.update(np.array([[False]]))

    assert tracker.get_counters()[0, 0] == 0
    assert not tracker.confirmed[0, 0]


def test_confirmation_clears_on_missing_frame():
    tracker = OccupancyPersistence((1, 1), persistence_frames=3)

    for _ in range(3):
        tracker.update(np.array([[True]]))

    assert tracker.confirmed[0, 0]

    tracker.update(np.array([[False]]))

    assert not tracker.confirmed[0, 0]


def test_counter_is_capped_at_three():
    tracker = OccupancyPersistence((1, 1), persistence_frames=3)

    for _ in range(10):
        tracker.update(np.array([[True]]))

    assert tracker.get_counters()[0, 0] == 3


def test_reset_clears_state():
    tracker = OccupancyPersistence((1, 1), persistence_frames=3)

    for _ in range(3):
        tracker.update(np.array([[True]]))

    tracker.reset()

    assert tracker.frame_count == 0
    assert tracker.get_counters()[0, 0] == 0
    assert not tracker.confirmed[0, 0]


def test_observation_grid_maps_points():
    config = {
        "mapping": {
            "x_min": 0,
            "x_max": 2,
            "y_min": 0,
            "y_max": 2,
        }
    }

    points = np.array([
        [0.25, 0.25, 0.0],
        [1.25, 1.25, 0.0],
        [3.00, 3.00, 0.0],
    ])

    grid = create_observation_grid(
        points, resolution=1.0, config=config
    )

    assert grid.shape == (2, 2)
    assert grid[0, 0]
    assert grid[1, 1]
    assert grid.sum() == 2


def test_invalid_observation_shape_rejected():
    tracker = OccupancyPersistence((2, 2), persistence_frames=3)

    with pytest.raises(ValueError):
        tracker.update(np.array([[True]]))
