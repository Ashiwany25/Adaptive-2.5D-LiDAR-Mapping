
"""
Semantic-aware resolution override and confidence gating.

Critical semantic classes may request fine resolution
even at longer distances.

Confidence gating prevents low-confidence semantic
predictions from automatically triggering refinement.

No semantic model is implemented in this module.
"""

import numpy as np

from lidar_mapping.mapping.adaptive_resolution import (
    select_distance_resolution,
)
from lidar_mapping.utils.config import load_config


def apply_semantic_override(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    confidence: np.ndarray | None = None,
    sensor_position: np.ndarray | None = None,
    config: dict | None = None,
) -> dict:
    """
    Compute semantic-aware point resolution requests.

    If confidence is None, confidence gating is disabled.
    This is useful for evaluating ground-truth labels.

    If confidence is provided, only critical points with
    confidence >= threshold trigger semantic refinement.

    Returns a dictionary containing point-level decisions.
    """

    if config is None:
        config = load_config()

    points = np.asarray(points)
    semantic_labels = np.asarray(semantic_labels)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    if semantic_labels.ndim != 1:
        raise ValueError(
            "Semantic labels must be a 1D array."
        )

    if len(points) != len(semantic_labels):
        raise ValueError(
            "Point and semantic label counts do not match."
        )

    if not np.issubdtype(
        semantic_labels.dtype, np.integer
    ):
        raise ValueError(
            "Semantic labels must contain integers."
        )

    if confidence is not None:
        confidence = np.asarray(
            confidence, dtype=np.float64
        )

        if confidence.shape != (len(points),):
            raise ValueError(
                "Confidence must have shape (N,)."
            )

        if not np.isfinite(confidence).all():
            raise ValueError(
                "Confidence must contain finite values."
            )

        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError(
                "Confidence must be between 0 and 1."
            )

    base_resolution = select_distance_resolution(
        points,
        sensor_position=sensor_position,
        config=config,
    )

    critical_classes = np.asarray(
        config["semantic"]["critical_classes"],
        dtype=np.int64,
    )

    critical_mask = np.isin(
        semantic_labels,
        critical_classes,
    )

    threshold = float(
        config["semantic"]["confidence_threshold"]
    )

    if confidence is None:
        confident_mask = np.ones(
            len(points), dtype=bool
        )
    else:
        confident_mask = confidence >= threshold

    override_eligible = (
        critical_mask & confident_mask
    )

    fine_resolution = float(
        config["mapping"]["minimum_resolution"]
    )

    final_resolution = base_resolution.copy()

    final_resolution[override_eligible] = np.minimum(
        final_resolution[override_eligible],
        fine_resolution,
    )

    # An actual override changes the requested resolution.
    override_applied = (
        final_resolution < base_resolution
    )

    return {
        "base_resolution": base_resolution,
        "final_resolution": final_resolution,
        "critical": critical_mask,
        "confident": confident_mask,
        "override_eligible": override_eligible,
        "override": override_applied,
    }


def summarize_semantic_override(
    result: dict,
) -> dict:
    """
    Summarize semantic refinement decisions.
    """

    base = np.asarray(result["base_resolution"])
    final = np.asarray(result["final_resolution"])
    critical = np.asarray(result["critical"])
    override = np.asarray(result["override"])

    return {
        "total_points": int(len(base)),
        "critical_points": int(critical.sum()),
        "actual_refined_points": int(override.sum()),
        "fine_points_before": int(
            np.sum(base == 0.25)
        ),
        "fine_points_after": int(
            np.sum(final == 0.25)
        ),
    }
