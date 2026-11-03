
"""
Evaluate locally generated SemanticKITTI pipeline results.

Reads:
    data/processed/real_pipeline_validation/

Creates:
    benchmarks/real_pipeline_evaluation/

Evaluation is based on 10 offline recorded frames.
No semantic inference or pose estimation is timed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_pipeline_validation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmarks"
    / "real_pipeline_evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(fig, filename):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=50, r=30, t=70, b=50),
    )

    fig.write_html(
        OUTPUT_DIR / filename,
        include_plotlyjs=True,
        full_html=True,
    )


def main():
    summary_path = INPUT_DIR / "pipeline_summary.csv"
    tracks_path = INPUT_DIR / "tracked_objects.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)

    summary = pd.read_csv(
        summary_path,
        dtype={"frame": str},
    )

    # An empty tracking CSV can have no columns if
    # no objects were tracked. Handle that case.
    try:
        tracks = pd.read_csv(
            tracks_path,
            dtype={"frame": str},
        )
    except pd.errors.EmptyDataError:
        tracks = pd.DataFrame()

    required = {
        "frame",
        "input_points",
        "static_points",
        "object_points",
        "unknown_points",
        "static_map_leaves",
        "object_map_leaves",
        "observed_static_atomic_cells",
        "confirmed_static_atomic_cells",
        "detected_instances",
        "observed_tracks",
        "processing_time_ms",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing summary columns: {sorted(missing)}"
        )

    assert len(summary) == 10, (
        "Expected exactly 10 evaluated frames."
    )

    assert summary["frame"].is_unique
    assert (
        summary["processing_time_ms"] >= 0
    ).all()

    assert (
        summary["static_points"]
        + summary["object_points"]
        + summary["unknown_points"]
        == summary["input_points"]
    ).all(), (
        "Semantic layer point counts do not "
        "sum to the input point count."
    )

    assert (
        summary["confirmed_static_atomic_cells"]
        <= summary["observed_static_atomic_cells"]
    ).all(), (
        "Confirmed cells exceed observed cells."
    )

    # Verify map files and mapped point conservation.
    map_checks = []

    for row in summary.itertuples(index=False):
        frame = row.frame

        static_map = pd.read_csv(
            INPUT_DIR / f"static_map_{frame}.csv"
        )

        object_map = pd.read_csv(
            INPUT_DIR / f"object_map_{frame}.csv"
        )

        assert len(static_map) == row.static_map_leaves
        assert len(object_map) == row.object_map_leaves

        static_mapped = int(
            static_map["point_count"].sum()
        )

        object_mapped = int(
            object_map["point_count"].sum()
        )

        assert static_mapped <= row.static_points
        assert object_mapped <= row.object_points

        temporal = np.load(
            INPUT_DIR / f"temporal_{frame}.npz"
        )

        observed = temporal["static_observed"]
        confirmed = temporal["static_confirmed"]

        assert observed.shape == confirmed.shape
        assert observed.dtype == np.bool_
        assert confirmed.dtype == np.bool_

        assert int(observed.sum()) == (
            row.observed_static_atomic_cells
        )

        assert int(confirmed.sum()) == (
            row.confirmed_static_atomic_cells
        )

        map_checks.append({
            "frame": frame,
            "static_mapped_points": static_mapped,
            "object_mapped_points": object_mapped,
            "total_mapped_points": (
                static_mapped + object_mapped
            ),
            "input_points": row.input_points,
        })

    map_checks_df = pd.DataFrame(map_checks)

    map_checks_df.to_csv(
        OUTPUT_DIR / "map_integrity_checks.csv",
        index=False,
    )

    print("Dataset and map integrity checks: PASSED")

    # 1. Processing time chart.
    fig_time = px.bar(
        summary,
        x="frame",
        y="processing_time_ms",
        title="Recorded LiDAR Pipeline Processing Time",
        labels={
            "frame": "Frame",
            "processing_time_ms": "Processing time (ms)",
        },
    )

    save_chart(
        fig_time,
        "processing_time_by_frame.html",
    )

    # 2. Map leaf count comparison.
    map_size = summary.melt(
        id_vars="frame",
        value_vars=[
            "static_map_leaves",
            "object_map_leaves",
        ],
        var_name="Layer",
        value_name="Occupied leaves",
    )

    fig_maps = px.line(
        map_size,
        x="frame",
        y="Occupied leaves",
        color="Layer",
        markers=True,
        title="Hierarchical Map Leaves by Frame",
    )

    save_chart(
        fig_maps,
        "hierarchical_map_leaves.html",
    )

    # 3. Temporal observation and confirmation.
    temporal_chart = summary.melt(
        id_vars="frame",
        value_vars=[
            "observed_static_atomic_cells",
            "confirmed_static_atomic_cells",
        ],
        var_name="Status",
        value_name="Atomic cells",
    )

    fig_temporal = px.line(
        temporal_chart,
        x="frame",
        y="Atomic cells",
        color="Status",
        markers=True,
        title="Static Observation Persistence (K=3)",
    )

    save_chart(
        fig_temporal,
        "temporal_confirmation.html",
    )

    # 4. Instance and tracking counts.
    fig_tracking = go.Figure()

    fig_tracking.add_trace(
        go.Scatter(
            x=summary["frame"],
            y=summary["detected_instances"],
            mode="lines+markers",
            name="Annotated instances",
        )
    )

    fig_tracking.add_trace(
        go.Scatter(
            x=summary["frame"],
            y=summary["observed_tracks"],
            mode="lines+markers",
            name="Observed tracks",
        )
    )

    fig_tracking.update_layout(
        title="Object Instance and Tracking Counts",
        xaxis_title="Frame",
        yaxis_title="Object count",
    )

    save_chart(
        fig_tracking,
        "object_tracking_counts.html",
    )

    # 5. Speed diagnostics, if tracks exist.
    speed_summary = {}

    if (
        not tracks.empty
        and "speed_mps" in tracks.columns
    ):
        valid_speeds = pd.to_numeric(
            tracks["speed_mps"],
            errors="coerce",
        ).dropna()

        if not valid_speeds.empty:
            fig_speed = px.histogram(
                x=valid_speeds,
                nbins=30,
                title="Estimated Tracked-Object Speeds",
                labels={
                    "x": "Estimated speed (m/s)",
                    "y": "Track observations",
                },
            )

            save_chart(
                fig_speed,
                "tracked_object_speeds.html",
            )

            speed_summary = {
                "mean_speed_mps": float(
                    valid_speeds.mean()
                ),
                "max_speed_mps": float(
                    valid_speeds.max()
                ),
            }

    metrics = {
        "dataset": "SemanticKITTI Sequence 00",
        "frame_count": int(len(summary)),
        "mean_processing_time_ms": float(
            summary["processing_time_ms"].mean()
        ),
        "median_processing_time_ms": float(
            summary["processing_time_ms"].median()
        ),
        "max_processing_time_ms": float(
            summary["processing_time_ms"].max()
        ),
        "mean_static_map_leaves": float(
            summary["static_map_leaves"].mean()
        ),
        "mean_object_map_leaves": float(
            summary["object_map_leaves"].mean()
        ),
        "total_instance_observations": int(
            summary["detected_instances"].sum()
        ),
        "total_track_observations": int(
            summary["observed_tracks"].sum()
        ),
        "mean_confirmed_static_atomic_cells": float(
            summary[
                "confirmed_static_atomic_cells"
            ].mean()
        ),
        "tracking_speed_diagnostics": speed_summary,
        "limitations": [
            "Only 10 consecutive recorded frames",
            "Ground-truth semantic and instance labels",
            "Ground-truth odometry pose alignment",
            "No semantic inference benchmark",
            "No live sensor acquisition",
            "No independent tracking accuracy score",
            "Processing time excludes data loading",
        ],
    }

    with (
        OUTPUT_DIR / "evaluation_metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nREAL-DATA EVALUATION SUMMARY")
    print("-" * 45)

    print(
        "Mean processing time:",
        f"{metrics['mean_processing_time_ms']:.2f} ms",
    )

    print(
        "Mean static map leaves:",
        f"{metrics['mean_static_map_leaves']:.1f}",
    )

    print(
        "Mean object map leaves:",
        f"{metrics['mean_object_map_leaves']:.1f}",
    )

    print(
        "Total instance observations:",
        metrics["total_instance_observations"],
    )

    print(
        "Total track observations:",
        metrics["total_track_observations"],
    )

    print("\nEvaluation charts saved to:")
    print(OUTPUT_DIR)

    print(
        "\nAll real-data evaluation checks passed!"
    )


if __name__ == "__main__":
    main()
