
"""
Distance-based adaptive resolution selection.

Selects the requested mapping resolution for
each LiDAR point based on sensor distance.

This module does not construct grid cells.
"""

import numpy as np

from lidar_mapping.utils.config import load_config


def calculate_distances(
    points: np.ndarray,
    sensor_position: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate 3D Euclidean distances from the sensor.

    Args:
        points: Array with shape (N, 3) or (N, 4+).
        sensor_position: Sensor XYZ position.

    Returns:
        Array of distances in meters.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    if not np.isfinite(points[:, :3]).all():
        raise ValueError(
            "Point coordinates must be finite."
        )

    if sensor_position is None:
        sensor_position = np.zeros(3, dtype=np.float64)

    sensor_position = np.asarray(
        sensor_position, dtype=np.float64
    )

    if sensor_position.shape != (3,):
        raise ValueError(
            "Sensor position must have shape (3,)."
        )

    if not np.isfinite(sensor_position).all():
        raise ValueError(
            "Sensor position must be finite."
        )

    return np.linalg.norm(
        points[:, :3] - sensor_position,
        axis=1,
    )


def select_distance_resolution(
    points: np.ndarray,
    sensor_position: np.ndarray | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """
    Assign a requested resolution to each point.

    Near: distance < 20 m
    Mid:  20 m <= distance < 40 m
    Far:  distance >= 40 m

    Thresholds and resolutions come from YAML.
    """
    if config is None:
        config = load_config()

    mapping = config["mapping"]

    near_distance = float(mapping["near_distance"])
    mid_distance = float(mapping["mid_distance"])

    near_resolution = float(mapping["near_resolution"])
    mid_resolution = float(mapping["mid_resolution"])
    far_resolution = float(mapping["far_resolution"])

    distances = calculate_distances(
        points, sensor_position
    )

    resolutions = np.full(
        len(distances),
        far_resolution,
        dtype=np.float64,
    )

    resolutions[
        distances < mid_distance
    ] = mid_resolution

    resolutions[
        distances < near_distance
    ] = near_resolution

    return resolutions


def summarize_resolution_distribution(
    resolutions: np.ndarray,
) -> dict[float, int]:
    """
    Count points assigned to each resolution.
    """
    resolutions = np.asarray(resolutions)

    if resolutions.ndim != 1:
        raise ValueError(
            "Resolutions must be a 1D array."
        )

    if not np.isfinite(resolutions).all():
        raise ValueError(
            "Resolutions must be finite."
        )

    if (resolutions <= 0).any():
        raise ValueError(
            "Resolutions must be positive."
        )

    values, counts = np.unique(
        resolutions,
        return_counts=True,
    )

    return {
        float(value): int(count)
        for value, count in zip(values, counts)
    }
