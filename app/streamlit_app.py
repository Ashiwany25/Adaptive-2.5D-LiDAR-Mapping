
"""
SIH 2026 - Adaptive 2.5D LiDAR Mapping Dashboard

Displays precomputed outputs from real SemanticKITTI
Sequence 00 frames.

This is offline recorded-data playback, not live
LiDAR streaming or autonomous vehicle operation.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lidar_mapping.visualization.adaptive_map import (
    create_adaptive_resolution_map,
    summarize_resolution_cells,
)

from lidar_mapping.visualization.comparison import (
    load_comparison_data,
    create_comparison_chart,
    calculate_relative_comparison,
)

import numpy as np

from lidar_mapping.visualization.temporal_tracking import (
    create_temporal_evidence,
    create_temporal_cell_map,
    create_track_trajectories,
    create_track_continuity,
    create_speed_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_pipeline_validation"
)


st.set_page_config(
    page_title="Adaptive 2.5D LiDAR Mapping",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data
def load_results():
    summary = pd.read_csv(
        RESULTS_DIR / "pipeline_summary.csv",
        dtype={"frame": str},
    )

    try:
        tracks = pd.read_csv(
            RESULTS_DIR / "tracked_objects.csv",
            dtype={"frame": str},
        )
    except pd.errors.EmptyDataError:
        tracks = pd.DataFrame()

    return summary, tracks


@st.cache_data
def load_frame_maps(frame_id):
    static_map = pd.read_csv(
        RESULTS_DIR / f"static_map_{frame_id}.csv"
    )

    object_map = pd.read_csv(
        RESULTS_DIR / f"object_map_{frame_id}.csv"
    )

    return static_map, object_map


def add_map_layer(fig, df, color, name):
    if df.empty:
        return

    x_values = []
    y_values = []

    for row in df.itertuples(index=False):
        x = float(row.x_min)
        y = float(row.y_min)
        size = float(row.resolution)

        x_values.extend([
            x, x + size, x + size,
            x, x, None
        ])

        y_values.extend([
            y, y, y + size,
            y + size, y, None
        ])

    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=y_values,
            mode="lines",
            line=dict(color=color, width=1),
            name=name,
            hoverinfo="skip",
        )
    )


def create_map(frame_id, static_map, object_map, tracks):
    fig = go.Figure()

    add_map_layer(
        fig, static_map, "#2684FF",
        "Static environment"
    )

    add_map_layer(
        fig, object_map, "#FF9F1C",
        "Object layer"
    )

    if not tracks.empty and "frame" in tracks.columns:
        frame_tracks = tracks[
            tracks["frame"] == frame_id
        ]

        if not frame_tracks.empty:
            fig.add_trace(
                go.Scatter(
                    x=frame_tracks["centroid_x"],
                    y=frame_tracks["centroid_y"],
                    mode="markers+text",
                    text=[
                        str(track_id)
                        for track_id in frame_tracks["track_id"]
                    ],
                    textposition="top center",
                    marker=dict(
                        color="red",
                        size=9,
                    ),
                    name="Tracked objects",
                )
            )

    fig.update_layout(
        title=f"Adaptive 2.5D Map - Frame {frame_id}",
        template="plotly_white",
        height=650,
        xaxis=dict(
            title="X (m)",
            range=[-50, 50],
        ),
        yaxis=dict(
            title="Y (m)",
            range=[-50, 50],
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h"),
    )

    return fig


def main():
    st.title("Adaptive 2.5D LiDAR Mapping")
    st.caption(
        "Smart India Hackathon 2026 | "
        "SemanticKITTI recorded-data demonstration"
    )

    st.info(
        "Offline playback using recorded LiDAR scans, "
        "dataset-provided semantic/instance annotations "
        "and recorded odometry poses. "
        "No live sensor or semantic inference."
    )

    if not (
        RESULTS_DIR / "pipeline_summary.csv"
    ).exists():
        st.error(
            "Precomputed pipeline results not found. "
            "Run scripts/run_real_pipeline.py first."
        )
        st.stop()

    summary, tracks = load_results()

    frames = summary["frame"].tolist()

    selected_frame = st.sidebar.selectbox(
        "Select LiDAR frame",
        frames,
        index=min(5, len(frames) - 1),
    )

    selected_row = summary.loc[
        summary["frame"] == selected_frame
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Input points",
        f"{int(selected_row['input_points']):,}",
    )

    col2.metric(
        "Static map leaves",
        f"{int(selected_row['static_map_leaves']):,}",
    )

    col3.metric(
        "Object map leaves",
        f"{int(selected_row['object_map_leaves']):,}",
    )

    col4.metric(
        "Observed tracks",
        int(selected_row["observed_tracks"]),
    )

    
    st.subheader("Interactive Adaptive Resolution Map")

    control1, control2, control3 = st.columns(3)

    with control1:
        show_static = st.checkbox(
            "Show static environment",
            value=True,
        )

    with control2:
        show_objects = st.checkbox(
            "Show object layer",
            value=True,
        )

    with control3:
        show_tracks = st.checkbox(
            "Show tracked objects",
            value=True,
        )

    static_map, object_map = load_frame_maps(
        selected_frame
    )

    if (
        not tracks.empty
        and "frame" in tracks.columns
    ):
        frame_tracks = tracks[
            tracks["frame"] == selected_frame
        ]
    else:
        frame_tracks = pd.DataFrame()

    fig = create_adaptive_resolution_map(
        static_map=static_map,
        object_map=object_map,
        tracks=frame_tracks,
        show_static=show_static,
        show_objects=show_objects,
        show_tracks=show_tracks,
        frame_id=selected_frame,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "Green: 0.25 m | Yellow: 0.50 m | "
        "Red: 1.00 m. "
        "Colors indicate resolution, not safety."
    )

    resolution_summary = summarize_resolution_cells(
        static_map,
        object_map,
    )

    st.subheader("Adaptive Resolution Distribution")

    fig_resolution = px.bar(
        resolution_summary,
        x="Resolution (m)",
        y="Occupied cells",
        color="Layer",
        barmode="group",
        title="Occupied Hierarchical Cells by Resolution",
    )

    st.plotly_chart(
        fig_resolution,
        use_container_width=True,
    )


    static_map, object_map = load_frame_maps(
        selected_frame
    )

    fig = create_map(
        selected_frame,
        static_map,
        object_map,
        tracks,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Recorded Pipeline Evaluation")

    left, right = st.columns(2)

    with left:
        fig_time = px.bar(
            summary,
            x="frame",
            y="processing_time_ms",
            title="Processing Time by Frame",
            labels={
                "frame": "Frame",
                "processing_time_ms": "Time (ms)",
            },
        )

        st.plotly_chart(
            fig_time,
            use_container_width=True,
        )

    with right:
        fig_leaves = px.line(
            summary,
            x="frame",
            y=[
                "static_map_leaves",
                "object_map_leaves",
            ],
            markers=True,
            title="Hierarchical Map Leaves",
        )

        st.plotly_chart(
            fig_leaves,
            use_container_width=True,
        )

    st.subheader("Temporal Observation Persistence")


    st.divider()

    st.subheader("Adaptive vs Uniform Mapping")

    st.caption(
        "Measured on 10 recorded SemanticKITTI frames. "
        "This comparison evaluates hierarchical map "
        "construction and elevation representation, "
        "not complete autonomous-driving performance."
    )

    comparison_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "comparison"
    )

    benchmark_file = (
        comparison_dir
        / "integrated_mapping_benchmark_summary.csv"
    )

    accuracy_file = (
        comparison_dir
        / "elevation_accuracy_summary.csv"
    )

    if benchmark_file.exists() and accuracy_file.exists():
        comparison_data = load_comparison_data(
            comparison_dir
        )

        changes = calculate_relative_comparison(
            comparison_data
        )

        st.dataframe(
            comparison_data,
            use_container_width=True,
            hide_index=True,
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Occupied-cell change vs uniform 0.25 m",
            (
                f"{changes['occupied_cells_change_pct']:+.1f}%"
                if pd.notna(changes["occupied_cells_change_pct"])
                else "N/A"
            ),
        )

        metric2.metric(
            "Build-time change vs uniform 0.25 m",
            (
                f"{changes['build_time_change_pct']:+.1f}%"
                if pd.notna(changes["build_time_change_pct"])
                else "N/A"
            ),
        )

        metric3.metric(
            "Elevation RMSE change vs uniform 0.25 m",
            (
                f"{changes['rmse_change_pct']:+.1f}%"
                if pd.notna(changes["rmse_change_pct"])
                else "N/A"
            ),
        )

        left_compare, right_compare = st.columns(2)

        with left_compare:
            st.plotly_chart(
                create_comparison_chart(
                    comparison_data,
                    "Average occupied cells",
                    "Occupied Map Cells",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                create_comparison_chart(
                    comparison_data,
                    "Average payload (KB)",
                    "Estimated Map Payload",
                ),
                use_container_width=True,
            )

        with right_compare:
            st.plotly_chart(
                create_comparison_chart(
                    comparison_data,
                    "Average build time (ms)",
                    "Map Construction Time",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                create_comparison_chart(
                    comparison_data,
                    "RMSE (m)",
                    "Elevation Reconstruction RMSE",
                ),
                use_container_width=True,
            )

        st.info(
            "RMSE is calculated from the same points used "
            "to construct each elevation map. It measures "
            "representation error, not obstacle-detection "
            "accuracy. Payload is an estimated numeric "
            "map representation size, not measured RAM."
        )

    else:
        st.warning(
            "Benchmark CSV files are missing from "
            "data/processed/comparison/."
        )


    fig_temporal = px.line(
        summary,
        x="frame",
        y=[
            "observed_static_atomic_cells",
            "confirmed_static_atomic_cells",
        ],
        markers=True,
        title="Observed vs Confirmed Static Cells (K=3)",
    )

    st.plotly_chart(
        fig_temporal,
        use_container_width=True,
    )

    st.subheader("Object Tracking")

    st.divider()

    st.header("Temporal Stability & Tracking Evidence")

    st.caption(
        "Results from 10 recorded, pose-aligned "
        "SemanticKITTI frames. Occupancy confirmation "
        "uses K=3 consecutive observations."
    )

    st.plotly_chart(
        create_temporal_evidence(summary),
        use_container_width=True,
    )

    temporal_file = (
        RESULTS_DIR
        / f"temporal_{selected_frame}.npz"
    )

    if temporal_file.exists():
        with np.load(temporal_file) as temporal_data:
            observed_grid = temporal_data[
                "static_observed"
            ].copy()

            confirmed_grid = temporal_data[
                "static_confirmed"
            ].copy()

        st.plotly_chart(
            create_temporal_cell_map(
                observed_grid,
                confirmed_grid,
                selected_frame,
            ),
            use_container_width=True,
        )

        st.caption(
            "Orange indicates currently observed cells "
            "that are not confirmed; green indicates "
            "confirmed cells. Confirmation is based on "
            "the temporal persistence rule, not "
            "independent ground-truth occupancy."
        )

    else:
        st.warning(
            f"Temporal data missing for frame {selected_frame}."
        )

    st.subheader("Multi-Object Tracking Evidence")

    if tracks.empty:
        st.info(
            "No object tracks are available in "
            "the recorded results."
        )

    else:
        trajectory_fig = create_track_trajectories(
            tracks
        )

        st.plotly_chart(
            trajectory_fig,
            use_container_width=True,
        )

        continuity_result = create_track_continuity(
            tracks
        )

        if continuity_result is not None:
            continuity_fig, continuity_table = (
                continuity_result
            )

            left_track, right_track = st.columns(2)

            with left_track:
                st.plotly_chart(
                    continuity_fig,
                    use_container_width=True,
                )

            with right_track:
                st.subheader("Track Continuity Table")

                st.dataframe(
                    continuity_table,
                    use_container_width=True,
                    hide_index=True,
                )

        speed_fig = create_speed_chart(tracks)

        if speed_fig is not None:
            st.plotly_chart(
                speed_fig,
                use_container_width=True,
            )

        st.caption(
            "Track trajectories and speed estimates "
            "come from the implemented association "
            "and constant-velocity tracking pipeline. "
            "They are not independently validated "
            "ground-truth object trajectories."
        )


    if tracks.empty:
        st.warning(
            "No tracked object records were generated."
        )
    else:
        frame_tracks = tracks[
            tracks["frame"] == selected_frame
        ]

        st.dataframe(
            frame_tracks,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Methodology and limitations"):
        st.markdown(
            """
            - **Dataset:** SemanticKITTI Sequence 00,
              10 consecutive recorded LiDAR frames.
            - **Alignment:** Dataset odometry poses
              and LiDAR-to-camera calibration.
            - **Semantics:** Ground-truth semantic
              and instance annotations.
            - **Resolution:** 0.25 m, 0.5 m, 1.0 m.
            - **Temporal confirmation:** K=3
              consecutive static observations.
            - **Tracking:** Class-gated Hungarian
              assignment and constant-velocity prediction.
            - **Runtime:** Local in-memory pipeline
              processing; excludes file loading,
              semantic inference and pose estimation.
            - **Limitations:** No live acquisition,
              no learned semantic predictions,
              no validated safety or real-time claims.
            """
        )

    st.caption(
        "Research prototype | "
        "Offline recorded-data playback"
    )


if __name__ == "__main__":
    main()
