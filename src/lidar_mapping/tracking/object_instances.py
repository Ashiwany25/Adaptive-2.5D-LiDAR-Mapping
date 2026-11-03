
"""
Extract annotated object instances from SemanticKITTI.

Uses semantic and instance labels to group LiDAR points.

This is ground-truth-assisted object extraction,
not an independently trained object detector.
"""

import numpy as np

from lidar_mapping.semantic.dynamic_separation import (
    POTENTIALLY_MOVABLE_IDS,
    ANNOTATED_MOVING_IDS,
)
from lidar_mapping.utils.config import load_config


OBJECT_CLASSES = (
    POTENTIALLY_MOVABLE_IDS | ANNOTATED_MOVING_IDS
)


# Normalize SemanticKITTI moving-class IDs to
# their corresponding base semantic categories.
BASE_CLASS = {
    252: 10,   # moving car
    253: 31,   # moving bicyclist
    254: 30,   # moving person
    255: 32,   # moving motorcyclist
    256: 20,   # moving on-rails
    257: 11,   # moving bus
    258: 18,   # moving truck
    259: 15,   # moving other-vehicle
}


def normalize_semantic_class(
    semantic_id: int,
) -> int:
    """
    Convert moving semantic ID to its base class.
    """
    semantic_id = int(semantic_id)
    return BASE_CLASS.get(semantic_id, semantic_id)


def extract_object_instances(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    instance_labels: np.ndarray,
    frame_id: int | str = 0,
    min_points: int | None = None,
) -> list[dict]:
    """
    Extract annotated object instances from one frame.

    Requirements:
    - Object belongs to a movable or moving class.
    - Instance ID is greater than zero.
    - At least min_points LiDAR returns are present.

    Returns:
        List of dictionaries describing objects.
    """
  
    
    if min_points is None:
        config = load_config()
        min_points = config["tracking"]["minimum_instance_points"]



    if (
        isinstance(min_points, (bool, np.bool_))
        or not isinstance(min_points, (int, np.integer))
        or min_points < 1
    ):
        raise ValueError(
            "min_points must be a positive integer."
        )

    points = np.asarray(points)
    semantic = np.asarray(semantic_labels)
    instance = np.asarray(instance_labels)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    if not np.isfinite(points[:, :3]).all():
        raise ValueError(
            "Point coordinates must be finite."
        )

    if semantic.shape != (len(points),):
        raise ValueError(
            "Semantic labels must have shape (N,)."
        )

    if instance.shape != (len(points),):
        raise ValueError(
            "Instance labels must have shape (N,)."
        )

    if not np.issubdtype(semantic.dtype, np.integer):
        raise ValueError(
            "Semantic labels must contain integers."
        )

    if not np.issubdtype(instance.dtype, np.integer):
        raise ValueError(
            "Instance labels must contain integers."
        )

    object_mask = (
        np.isin(semantic, list(OBJECT_CLASSES))
        & (instance > 0)
    )

    object_points = points[object_mask, :3]
    object_semantic = semantic[object_mask]
    object_instance = instance[object_mask]

    objects = []

    for instance_id in np.unique(object_instance):
        mask = object_instance == instance_id

        xyz = object_points[mask]
        labels = object_semantic[mask]

        if len(xyz) < min_points:
            continue

        # Use the dominant original semantic label.
        class_ids, counts = np.unique(
            labels,
            return_counts=True,
        )

        dominant_class = int(
            class_ids[np.argmax(counts)]
        )

        centroid = xyz.mean(axis=0)
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)

        objects.append({
            "frame_id": frame_id,
            "instance_id": int(instance_id),
            "semantic_id": dominant_class,
            "base_semantic_id": normalize_semantic_class(
                dominant_class
            ),
            "point_count": int(len(xyz)),
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": bbox_max - bbox_min,
            "points": xyz.copy(),
        })

    return objects


def summarize_object_instances(
    objects: list[dict],
) -> dict:
    """
    Summarize the extracted object records.
    """

    return {
        "object_count": len(objects),
        "total_object_points": sum(
            obj["point_count"] for obj in objects
        ),
        "unique_base_classes": len({
            obj["base_semantic_id"]
            for obj in objects
        }),
    }
