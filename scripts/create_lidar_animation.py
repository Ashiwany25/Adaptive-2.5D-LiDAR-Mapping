
"""
Animated 2.5D LiDAR map using 10 real
SemanticKITTI Sequence 00 frames.

This is offline recorded-data playback,
not live LiDAR streaming.
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
    / "data/raw/SemanticKITTI/sequences/00"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data/processed/real_pipeline_validation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def map_to_rectangles(df):
    """
    Convert hierarchical grid leaves to
    rectangle boundary coordinates.
    """
    xs = []
    ys = []

    for row in df.itertuples(index=False):
        x0 = float(row.x_min)
        y0 = float(row.y_min)
        size = float(row.resolution)

        xs.extend([
            x0, x0 + size, x0 + size,
            x0, x0, None
        ])

        ys.extend([
            y0, y0, y0 + size,
            y0 + size, y0, None
        ])

    return xs, ys


def make_frame_traces(
    frame_id,
    sensor_position,
    tracks,
):
    static_map = pd.read_csv(
        RESULTS_DIR / f"static_map_{frame_id}.csv"
    )

    object_map = pd.read_csv(
        RESULTS_DIR / f"object_map_{frame_id}.csv"
    )

    static_x, static_y = map_to_rectangles(static_map)
    object_x, object_y = map_to_rectangles(object_map)

    if (
        not tracks.empty
        and "frame" in tracks.columns
    ):
        frame_tracks = tracks[
            tracks["frame"] == frame_id
        ]
    else:
        frame_tracks = pd.DataFrame()

    if frame_tracks.empty:
        track_x = []
        track_y = []
        track_text = []
    else:
        track_x = frame_tracks["centroid_x"].tolist()
        track_y = frame_tracks["centroid_y"].tolist()
        track_text = [
            f"ID {track_id}"
            for track_id in frame_tracks["track_id"]
        ]

    return [
        go.Scattergl(
            x=static_x,
            y=static_y,
            mode="lines",
            line=dict(
                color="#2684FF",
                width=1,
            ),
            name="Static map",
            hoverinfo="skip",
        ),
        go.Scattergl(
            x=object_x,
            y=object_y,
            mode="lines",
            line=dict(
                color="#FF9F1C",
                width=1.5,
            ),
            name="Object layer",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=track_x,
            y=track_y,
            text=track_text,
            mode="markers+text",
            textposition="top center",
            marker=dict(
                color="red",
                size=9,
            ),
            name="Tracked objects",
        ),
        go.Scatter(
            x=[sensor_position[0]],
            y=[sensor_position[1]],
            mode="markers",
            marker=dict(
                color="black",
                size=12,
                symbol="diamond",
            ),
            name="LiDAR sensor",
        ),
    ]


def main():
    frames, _ = load_aligned_semantickitti_frames(
        DATA_ROOT,
        frame_count=10,
    )

    tracks_path = (
        RESULTS_DIR / "tracked_objects.csv"
    )

    try:
        tracks = pd.read_csv(
            tracks_path,
            dtype={"frame": str},
        )
    except pd.errors.EmptyDataError:
        tracks = pd.DataFrame()

    animation_frames = []

    for frame in frames:
        frame_id = frame["frame_id"]

        traces = make_frame_traces(
            frame_id,
            frame["sensor_position"],
            tracks,
        )

        animation_frames.append(
            go.Frame(
                data=traces,
                name=frame_id,
            )
        )

        print(f"Prepared frame {frame_id}")

    fig = go.Figure(
        data=animation_frames[0].data,
        frames=animation_frames,
    )

    slider_steps = []

    for frame in frames:
        frame_id = frame["frame_id"]

        slider_steps.append({
            "args": [
                [frame_id],
                {
                    "frame": {
                        "duration": 0,
                        "redraw": True,
                    },
                    "mode": "immediate",
                    "transition": {
                        "duration": 0,
                    },
                },
            ],
            "label": frame_id,
            "method": "animate",
        })

    fig.update_layout(
        title=(
            "Adaptive 2.5D LiDAR Mapping — "
            "SemanticKITTI Recorded Playback"
        ),
        template="plotly_white",
        width=1100,
        height=850,
        xaxis=dict(
            title="X (meters)",
            range=[-50, 50],
        ),
        yaxis=dict(
            title="Y (meters)",
            range=[-50, 50],
            scaleanchor="x",
            scaleratio=1,
        ),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "showactive": False,
            "x": 0.05,
            "y": -0.12,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {
                                "duration": 600,
                                "redraw": True,
                            },
                            "fromcurrent": True,
                            "transition": {
                                "duration": 0,
                            },
                        },
                    ],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {
                                "duration": 0,
                                "redraw": False,
                            },
                            "mode": "immediate",
                        },
                    ],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "currentvalue": {
                "prefix": "Frame: ",
            },
            "pad": {
                "t": 50,
            },
            "steps": slider_steps,
        }],
        margin=dict(
            l=60,
            r=30,
            t=80,
            b=130,
        ),
    )

    output_path = (
        OUTPUT_DIR / "lidar_animation_10_frames.html"
    )

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )

    print(
        "\nInteractive LiDAR animation created!"
    )
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
