
"""
SemanticKITTI / KITTI odometry pose alignment.

KITTI odometry poses describe the camera coordinate
frame in a common world coordinate system.

The calibration matrix Tr transforms LiDAR coordinates
into camera coordinates.

For frame i:
    T_world_lidar_i = T_world_camera_i @ T_camera_lidar

To express every scan in LiDAR frame 0:
    T_lidar0_lidari =
        inverse(T_world_lidar_0) @ T_world_lidar_i
"""

from pathlib import Path

import numpy as np


def load_kitti_poses(path):
    """
    Read KITTI odometry poses.txt.

    Each line contains 12 numbers representing
    a 3x4 transformation matrix.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    poses = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            values = np.fromstring(
                line, sep=" ", dtype=np.float64
            )

            if values.size != 12:
                raise ValueError(
                    f"Invalid pose on line {line_number}: "
                    f"expected 12 values, got {values.size}"
                )

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :] = values.reshape(3, 4)

            if not np.isfinite(transform).all():
                raise ValueError(
                    f"Nonfinite pose on line {line_number}"
                )

            poses.append(transform)

    if not poses:
        raise ValueError("No poses found.")

    return np.stack(poses)


def load_kitti_calibration(path):
    """
    Read the LiDAR-to-camera transform 'Tr'
    from KITTI odometry calib.txt.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, values_text = line.split(":", 1)

            if key.strip() != "Tr":
                continue

            values = np.fromstring(
                values_text, sep=" ", dtype=np.float64
            )

            if values.size != 12:
                raise ValueError(
                    "Calibration Tr must contain 12 values."
                )

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :] = values.reshape(3, 4)

            if not np.isfinite(transform).all():
                raise ValueError(
                    "Calibration contains nonfinite values."
                )

            return transform

    raise ValueError(
        "Calibration file does not contain 'Tr'."
    )


def calculate_relative_lidar_poses(
    camera_poses,
    lidar_to_camera,
):
    """
    Return transformations from each LiDAR frame
    into the first LiDAR frame.
    """
    camera_poses = np.asarray(
        camera_poses, dtype=np.float64
    )
    lidar_to_camera = np.asarray(
        lidar_to_camera, dtype=np.float64
    )

    if (
        camera_poses.ndim != 3
        or camera_poses.shape[1:] != (4, 4)
    ):
        raise ValueError(
            "Camera poses must have shape (N, 4, 4)."
        )

    if lidar_to_camera.shape != (4, 4):
        raise ValueError(
            "LiDAR calibration must have shape (4, 4)."
        )

    world_lidar_poses = (
        camera_poses @ lidar_to_camera
    )

    first_inverse = np.linalg.inv(
        world_lidar_poses[0]
    )

    relative_poses = (
        first_inverse[None, :, :]
        @ world_lidar_poses
    )

    return relative_poses


def align_lidar_points(points, transform):
    """
    Apply a 4x4 rigid transformation.

    Preserves optional intensity/extra columns.
    """
    points = np.asarray(points)
    transform = np.asarray(
        transform, dtype=np.float64
    )

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3+)."
        )

    if transform.shape != (4, 4):
        raise ValueError(
            "Transform must have shape (4, 4)."
        )

    aligned = points.copy()

    xyz = np.asarray(
        points[:, :3], dtype=np.float64
    )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    aligned_xyz = (
        xyz @ rotation.T + translation
    )

    aligned = aligned.astype(
        np.float64, copy=False
    )
    aligned[:, :3] = aligned_xyz

    return aligned


def load_aligned_semantickitti_frames(
    sequence_root,
    frame_count=10,
):
    """
    Load the first N scans and labels and align
    their XYZ coordinates into LiDAR frame 0.

    Returns:
        frames: list of dictionaries
        relative_poses: (number_of_poses, 4, 4)
    """
    sequence_root = Path(sequence_root)

    camera_poses = load_kitti_poses(
        sequence_root / "poses.txt"
    )

    lidar_to_camera = load_kitti_calibration(
        sequence_root / "calib.txt"
    )

    relative_poses = calculate_relative_lidar_poses(
        camera_poses,
        lidar_to_camera,
    )

    if frame_count < 1:
        raise ValueError(
            "frame_count must be at least 1."
        )

    if frame_count > len(relative_poses):
        raise ValueError(
            "Not enough recorded poses."
        )

    frames = []

    for frame_index in range(frame_count):
        frame_name = f"{frame_index:06d}"

        scan_path = (
            sequence_root
            / "velodyne"
            / f"{frame_name}.bin"
        )

        label_path = (
            sequence_root
            / "labels"
            / f"{frame_name}.label"
        )

        raw_scan = np.fromfile(
            scan_path, dtype=np.float32
        ).reshape(-1, 4)

        raw_labels = np.fromfile(
            label_path, dtype=np.uint32
        )

        if len(raw_scan) != len(raw_labels):
            raise ValueError(
                f"Point/label mismatch: {frame_name}"
            )

        semantic = (
            raw_labels & 0xFFFF
        ).astype(np.uint16)

        instance = (
            raw_labels >> 16
        ).astype(np.uint16)

        aligned = align_lidar_points(
            raw_scan,
            relative_poses[frame_index],
        )

        frames.append({
            "frame_id": frame_name,
            "points": aligned[:, :3],
            "intensity": raw_scan[:, 3].copy(),
            "semantic": semantic,
            "instance": instance,
            "sensor_position": (
                relative_poses[frame_index][:3, 3].copy()
            ),
        })

    return frames, relative_poses
