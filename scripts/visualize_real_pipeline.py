
"""
Interactive visualization of real SemanticKITTI
2.5D LiDAR perception results.

Creates one HTML visualization per recorded frame.

Uses:
- Static hierarchical elevation maps
- Object hierarchical elevation maps
- Recorded object tracking results

All visualizations are offline recorded-data playback.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_pipeline_validation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_elevation_cells(
    fig,
    dataframe,
    name,
    color,
    opacity=0.65,
):
    """
    Draw occupied hierarchical cells as rectangles
    in the XY plane.

    Hover displays the cell's elevation statistics.
    """

    if dataframe.empty:
        return

    xs = []
    ys = []

    for row in dataframe.itertuples(index=False):
        x0 = float(row.x_min)
        y0 = float(row.y_min)
        resolution = float(row.resolution)

        x1 = x0 + resolution
        y1 = y0 + resolution

        xs.extend([
            x0, x1, x1, x0, x0, None
        ])

        ys.extend([
            y0, y0, y1, y1, y0, None
        ])

    fig.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color=color,
                width=1,
            ),
            opacity=opacity,
            name=name,
            hoverinfo="skip",
        )
    )

    # Cell centers provide hover information.
    centers_x = (
        dataframe["x_min"]
        + dataframe["resolution"] / 2
    )

    centers_y = (
        dataframe["y_min"]
        + dataframe["resolution"] / 2
    )

    custom = np.column_stack([
        dataframe["resolution"],
        dataframe["point_count"],
        dataframe["z_min"],
        dataframe["z_max"],
        dataframe["z_mean"],
    ])

    fig.add_trace(
        go.Scattergl(
            x=centers_x,
            y=centers_y,
            mode="markers",
            marker=dict(
                size=3,
                color=color,
                opacity=0.5,
            ),
            customdata=custom,
            hovertemplate=(
                "X: %{x:.2f} m"
                "<br>Y: %{y:.2f} m"
                "<br>Resolution: %{customdata[0]:.2f} m"
                "<br>Points: %{customdata[1]:.0f}"
                "<br>Z min: %{customdata[2]:.2f} m"
                "<br>Z max: %{customdata[3]:.2f} m"
                "<br>Z mean: %{customdata[4]:.2f} m"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )


def add_tracks(fig, tracks):
    if tracks.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=tracks["centroid_x"],
            y=tracks["centroid_y"],
            mode="markers+text",
            text=[
                f"ID {track_id}"
                for track_id in tracks["track_id"]
            ],
            textposition="top center",
            marker=dict(
                size=9,
                color="red",
                symbol="circle",
            ),
            name="Tracked objects",
            hovertemplate=(
                "Track ID: %{text}"
                "<br>X: %{x:.2f} m"
                "<br>Y: %{y:.2f} m"
                "<extra></extra>"
            ),
        )
    )


def main():
    frames, relative_poses = (
        load_aligned_semantickitti_frames(
            DATA_ROOT,
            frame_count=10,
        )
    )

    tracks_path = INPUT_DIR / "tracked_objects.csv"

    try:
        all_tracks = pd.read_csv(
            tracks_path,
            dtype={"frame": str},
        )
    except pd.errors.EmptyDataError:
        all_tracks = pd.DataFrame()

    for frame in frames:
        frame_id = frame["frame_id"]

        static_map = pd.read_csv(
            INPUT_DIR / f"static_map_{frame_id}.csv"
        )

        object_map = pd.read_csv(
            INPUT_DIR / f"object_map_{frame_id}.csv"
        )

        fig = go.Figure()

        add_elevation_cells(
            fig,
            static_map,
            "Static environment",
            "#2684FF",
            opacity=0.65,
        )

        add_elevation_cells(
            fig,
            object_map,
            "Object layer",
            "#FF9F1C",
            opacity=0.85,
        )

        if (
            not all_tracks.empty
            and "frame" in all_tracks.columns
        ):
            frame_tracks = all_tracks[
                all_tracks["frame"] == frame_id
            ]

            add_tracks(fig, frame_tracks)

        sensor = frame["sensor_position"]

        fig.add_trace(
            go.Scatter(
                x=[sensor[0]],
                y=[sensor[1]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=12,
                    color="black",
                ),
                name="LiDAR sensor",
            )
        )

        fig.update_layout(
            title=(
                f"2.5D Adaptive LiDAR Mapping — "
                f"SemanticKITTI Frame {frame_id}"
            ),
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            template="plotly_white",
            width=1100,
            height=850,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        output_path = (
            OUTPUT_DIR
            / f"adaptive_map_{frame_id}.html"
        )

        fig.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
        )

        print(
            f"Created: {output_path.name} | "
            f"Static leaves: {len(static_map):,} | "
            f"Object leaves: {len(object_map):,}"
        )

    print(
        "\nAll 10 interactive LiDAR maps generated!"
    )
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
