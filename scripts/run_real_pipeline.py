
"""
Run the integrated 2.5D LiDAR pipeline on
10 pose-aligned SemanticKITTI Sequence 00 frames.

This is offline recorded-data evaluation.

Semantic and instance annotations are provided
by SemanticKITTI, not predicted by a model.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from lidar_mapping.pipeline import LidarPerceptionPipeline
from lidar_mapping.preprocessing.pose_alignment import (
    load_aligned_semantickitti_frames,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "SemanticKITTI"
    / "sequences"
    / "00"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_pipeline_validation"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def save_map(dataframe, path):
    if isinstance(dataframe, pd.DataFrame):
        dataframe.to_csv(path, index=False)
    else:
        raise TypeError(
            "Expected a pandas DataFrame for map output."
        )


def main():
    print("Loading 10 real SemanticKITTI frames...")

    frames, relative_poses = (
        load_aligned_semantickitti_frames(
            DATA_ROOT,
            frame_count=10,
        )
    )

    pipeline = LidarPerceptionPipeline()

    summary_rows = []
    tracking_rows = []

    for frame in frames:
        start = time.perf_counter()

        result = pipeline.process_frame(
            points=frame["points"],
            semantic_labels=frame["semantic"],
            instance_labels=frame["instance"],
            sensor_position=frame["sensor_position"],
            confidence=None,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        frame_id = frame["frame_id"]

        static_map = result["static_map"]
        object_map = result["object_map"]

        # Verify point conservation inside mapping bounds.
        config = pipeline.config["mapping"]
        points = frame["points"]

        valid_bounds = (
            (points[:, 0] >= config["x_min"])
            & (points[:, 0] < config["x_max"])
            & (points[:, 1] >= config["y_min"])
            & (points[:, 1] < config["y_max"])
        )

        mapped_points = (
            int(static_map["point_count"].sum())
            + int(object_map["point_count"].sum())
        )

        # Unknown semantic classes are not included
        # in the static/object map layers.
        assert mapped_points <= int(
            valid_bounds.sum()
        ), (
            f"Mapped point count exceeds input "
            f"for frame {frame_id}"
        )

        confirmed_cells = int(
            result["static_confirmed"].sum()
        )

        observed_cells = int(
            result["static_observed"].sum()
        )

        summary_rows.append({
            "frame": frame_id,
            "input_points": len(points),
            "static_points": (
                result["static_point_count"]
            ),
            "object_points": (
                result["object_point_count"]
            ),
            "unknown_points": (
                result["unknown_point_count"]
            ),
            "static_map_leaves": len(static_map),
            "object_map_leaves": len(object_map),
            "observed_static_atomic_cells": (
                observed_cells
            ),
            "confirmed_static_atomic_cells": (
                confirmed_cells
            ),
            "detected_instances": len(
                result["objects"]
            ),
            "observed_tracks": len(
                result["tracks"]
            ),
            "processing_time_ms": elapsed_ms,
        })

        for track in result["tracks"]:
            centroid = track["centroid"]
            velocity = track["velocity"]

            tracking_rows.append({
                "frame": frame_id,
                "track_id": track["track_id"],
                "instance_id": track["instance_id"],
                "semantic_id": (
                    track["base_semantic_id"]
                ),
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "centroid_z": centroid[2],
                "velocity_x": velocity[0],
                "velocity_y": velocity[1],
                "velocity_z": velocity[2],
                "speed_mps": track["speed_mps"],
                "association": track["association"],
            })

        save_map(
            static_map,
            OUTPUT_ROOT
            / f"static_map_{frame_id}.csv",
        )

        save_map(
            object_map,
            OUTPUT_ROOT
            / f"object_map_{frame_id}.csv",
        )

        np.savez_compressed(
            OUTPUT_ROOT
            / f"temporal_{frame_id}.npz",
            static_observed=(
                result["static_observed"]
            ),
            static_confirmed=(
                result["static_confirmed"]
            ),
        )

        print(
            f"Frame {frame_id} | "
            f"Points: {len(points):,} | "
            f"Static leaves: {len(static_map):,} | "
            f"Object leaves: {len(object_map):,} | "
            f"Tracks: {len(result['tracks'])} | "
            f"Time: {elapsed_ms:.2f} ms"
        )

    summary_df = pd.DataFrame(summary_rows)
    tracking_df = pd.DataFrame(tracking_rows)

    summary_df.to_csv(
        OUTPUT_ROOT / "pipeline_summary.csv",
        index=False,
    )

    tracking_df.to_csv(
        OUTPUT_ROOT / "tracked_objects.csv",
        index=False,
    )

    metadata = {
        "dataset": "SemanticKITTI",
        "sequence": "00",
        "frames": len(frames),
        "processing_mode": "offline_recorded_data",
        "pose_source": "KITTI odometry poses",
        "semantic_source": (
            "SemanticKITTI ground-truth labels"
        ),
        "instance_source": (
            "SemanticKITTI ground-truth labels"
        ),
        "confidence_source": None,
        "mapping": "adaptive hierarchical 2.5D",
        "temporal_persistence_frames": 3,
        "runtime_scope": (
            "in-memory pipeline processing only; "
            "excludes scan loading and pose estimation"
        ),
    }

    with (
        OUTPUT_ROOT / "validation_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    assert len(summary_df) == 10

    print("\nValidation summary:")
    print(summary_df.to_string(index=False))

    print(
        "\nMean processing time: "
        f"{summary_df['processing_time_ms'].mean():.2f} ms"
    )

    print(
        "Total recorded track observations: "
        f"{len(tracking_df)}"
    )

    print(
        "\nResults saved to:",
        OUTPUT_ROOT,
    )

    print(
        "\nAll 10 real SemanticKITTI frames "
        "processed successfully!"
    )


if __name__ == "__main__":
    main()
