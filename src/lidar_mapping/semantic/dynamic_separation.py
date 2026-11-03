
"""
SemanticKITTI static and dynamic layer separation.

Uses ground-truth semantic and instance annotations.

Important:
- Potentially movable does not mean currently moving.
- Moving-class annotations are dataset labels.
- This module does not estimate motion from LiDAR.
"""

import numpy as np


# SemanticKITTI original semantic IDs.
STATIC_ENVIRONMENT_IDS = {
    40, 44, 48, 49, 50, 51, 52,
    60, 70, 71, 72, 80, 81,
}

POTENTIALLY_MOVABLE_IDS = {
    10, 11, 13, 15, 16, 18,
    20, 30, 31, 32,
}

ANNOTATED_MOVING_IDS = {
    252, 253, 254, 255,
    256, 257, 258, 259,
}


def separate_semantic_layers(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    instance_labels: np.ndarray | None = None,
) -> dict:
    """
    Separate point cloud into semantic layers.

    Returns masks and point subsets for:
    - static environment
    - potentially movable objects
    - annotated moving objects
    - unknown or other points

    Instance IDs are retained for later tracking.
    """

    points = np.asarray(points)
    semantic = np.asarray(semantic_labels)

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

    if not np.issubdtype(semantic.dtype, np.integer):
        raise ValueError(
            "Semantic labels must contain integers."
        )

    if instance_labels is None:
        instance = np.zeros(
            len(points), dtype=np.uint16
        )
    else:
        instance = np.asarray(instance_labels)

        if instance.shape != (len(points),):
            raise ValueError(
                "Instance labels must have shape (N,)."
            )

        if not np.issubdtype(
            instance.dtype, np.integer
        ):
            raise ValueError(
                "Instance labels must contain integers."
            )

    static_mask = np.isin(
        semantic,
        list(STATIC_ENVIRONMENT_IDS),
    )

    movable_mask = np.isin(
        semantic,
        list(POTENTIALLY_MOVABLE_IDS),
    )

    moving_mask = np.isin(
        semantic,
        list(ANNOTATED_MOVING_IDS),
    )

    known_mask = (
        static_mask | movable_mask | moving_mask
    )

    unknown_mask = ~known_mask

    # The object layer includes both potentially
    # movable and annotated-moving objects.
    object_mask = movable_mask | moving_mask

    return {
        "static_mask": static_mask,
        "movable_mask": movable_mask,
        "moving_mask": moving_mask,
        "unknown_mask": unknown_mask,
        "object_mask": object_mask,

        "static_points": points[static_mask],
        "movable_points": points[movable_mask],
        "moving_points": points[moving_mask],
        "unknown_points": points[unknown_mask],
        "object_points": points[object_mask],

        "static_semantic": semantic[static_mask],
        "object_semantic": semantic[object_mask],
        "object_instance": instance[object_mask],
    }


def summarize_semantic_layers(
    result: dict,
) -> dict:
    """
    Return point counts for each semantic category.
    """

    static_count = int(
        result["static_mask"].sum()
    )

    movable_count = int(
        result["movable_mask"].sum()
    )

    moving_count = int(
        result["moving_mask"].sum()
    )

    unknown_count = int(
        result["unknown_mask"].sum()
    )

    total = (
        static_count
        + movable_count
        + moving_count
        + unknown_count
    )

    return {
        "total_points": total,
        "static_points": static_count,
        "potentially_movable_points": movable_count,
        "annotated_moving_points": moving_count,
        "unknown_points": unknown_count,
        "object_layer_points": int(
            result["object_mask"].sum()
        ),
    }
