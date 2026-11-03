import numpy as np

from lidar_mapping.mapping.adaptive_resolution import (
    calculate_distances,
    select_distance_resolution,
)


CONFIG = {
    "mapping": {
        "near_distance": 20.0,
        "mid_distance": 40.0,
        "near_resolution": 0.25,
        "mid_resolution": 0.5,
        "far_resolution": 1.0,
    }
}


def test_distance_calculation():
    points = np.array([
        [3.0, 4.0, 0.0],
        [6.0, 8.0, 0.0],
    ])

    sensor = np.array([0.0, 0.0, 0.0])

    distances = calculate_distances(points, sensor)

    np.testing.assert_allclose(distances, [5.0, 10.0])


def test_near_resolution():
    points = np.array([[10.0, 0.0, 0.0]])
    sensor = np.zeros(3)

    result = select_distance_resolution(points, sensor, CONFIG)

    np.testing.assert_allclose(result, [0.25])


def test_mid_resolution():
    points = np.array([[30.0, 0.0, 0.0]])
    sensor = np.zeros(3)

    result = select_distance_resolution(points, sensor, CONFIG)

    np.testing.assert_allclose(result, [0.5])


def test_far_resolution():
    points = np.array([[45.0, 0.0, 0.0]])
    sensor = np.zeros(3)

    result = select_distance_resolution(points, sensor, CONFIG)

    np.testing.assert_allclose(result, [1.0])


def test_sensor_relative_distance():
    points = np.array([[110.0, 0.0, 0.0]])
    sensor = np.array([100.0, 0.0, 0.0])

    distances = calculate_distances(points, sensor)

    np.testing.assert_allclose(distances, [10.0])
