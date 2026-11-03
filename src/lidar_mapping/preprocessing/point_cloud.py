
"""
SemanticKITTI point cloud preprocessing.

Supports:
- LiDAR scan loading
- Semantic and instance label decoding
- Input validation
- Coordinate transformations
- Mapping boundary filtering
"""

from pathlib import Path

import numpy as np


def load_lidar_scan(file_path: str | Path) -> np.ndarray:
    """
    Load a SemanticKITTI LiDAR scan.

    Each point contains:
        x, y, z, remission

    Returns:
        Array of shape (N, 4).
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"LiDAR scan not found: {path}"
        )

    raw = np.fromfile(path, dtype=np.float32)

    if raw.size % 4 != 0:
        raise ValueError(
            "Invalid LiDAR scan: expected 4 values per point."
        )

    points = raw.reshape(-1, 4)

    if not np.isfinite(points).all():
        raise ValueError(
            "LiDAR scan contains NaN or infinite values."
        )

    return points


def load_semantic_labels(
    file_path: str | Path,
    expected_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode SemanticKITTI packed uint32 labels.

    Lower 16 bits: semantic class
    Upper 16 bits: instance ID
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Semantic label file not found: {path}"
        )

    packed = np.fromfile(path, dtype=np.uint32)

    if (
        expected_points is not None
        and len(packed) != expected_points
    ):
        raise ValueError(
            "Point and label counts do not match."
        )

    semantic = (
        packed & np.uint32(0xFFFF)
    ).astype(np.uint16)

    instance = (
        packed >> np.uint32(16)
    ).astype(np.uint16)

    return semantic, instance


def load_labeled_frame(
    scan_path: str | Path,
    label_path: str | Path,
) -> dict:
    """
    Load one LiDAR frame and its corresponding labels.
    """
    points = load_lidar_scan(scan_path)

    semantic, instance = load_semantic_labels(
        label_path,
        expected_points=len(points),
    )

    return {
        "points": points,
        "semantic": semantic,
        "instance": instance,
    }


def transform_points(
    points: np.ndarray,
    transformation: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D coordinates using a 4x4 matrix.

    The transformation must map source coordinates
    into the desired target coordinate system.

    Remission and other extra columns are preserved.
    """
    points = np.asarray(points)
    transformation = np.asarray(transformation)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    if transformation.shape != (4, 4):
        raise ValueError(
            "Transformation must have shape (4, 4)."
        )

    if not np.isfinite(points).all():
        raise ValueError("Points contain non-finite values.")

    if not np.isfinite(transformation).all():
        raise ValueError(
            "Transformation contains non-finite values."
        )

    xyz = points[:, :3]

    rotation = transformation[:3, :3]
    translation = transformation[:3, 3]

    transformed_xyz = (
        xyz @ rotation.T + translation
    )

    result = points.copy()
    result[:, :3] = transformed_xyz

    return result


def filter_mapping_bounds(
    points: np.ndarray,
    x_min: float = -50.0,
    x_max: float = 50.0,
    y_min: float = -50.0,
    y_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep finite points within the XY mapping region.

    Returns:
        filtered_points
        original_boolean_mask
    """
    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have at least 3 columns."
        )

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid mapping boundaries.")

    mask = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )

    return points[mask], mask
