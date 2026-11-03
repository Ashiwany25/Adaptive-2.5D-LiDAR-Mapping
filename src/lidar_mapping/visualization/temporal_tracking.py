
"""
Temporal stability and object-tracking visualizations.

Uses previously computed SemanticKITTI results.
No synthetic benchmark values are introduced.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_temporal_evidence(summary):
    required = {
        "frame",
        "observed_static_atomic_cells",
        "confirmed_static_atomic_cells",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            f"Missing temporal columns: {sorted(missing)}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=summary["frame"],
            y=summary["observed_static_atomic_cells"],
            mode="lines+markers",
            name="Observed static cells",
            line=dict(color="#F59E0B", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=summary["frame"],
            y=summary["confirmed_static_atomic_cells"],
            mode="lines+markers",
            name="K=3 confirmed cells",
            line=dict(color="#16A34A", width=3),
        )
    )

    fig.update_layout(
        title="Static Occupancy Observation vs Confirmation",
        xaxis_title="Recorded frame",
        yaxis_title="Atomic grid cells",
        template="plotly_white",
        height=420,
    )

    return fig


def create_temporal_cell_map(
    observed,
    confirmed,
    frame_id,
):
    observed = np.asarray(observed, dtype=bool)
    confirmed = np.asarray(confirmed, dtype=bool)

    if observed.shape != confirmed.shape:
        raise ValueError(
            "Observed and confirmed grid shapes differ."
        )

    # Three categories:
    # 0 = no observed or confirmed occupancy
    # 1 = observed but not confirmed
    # 2 = confirmed occupancy
    display = np.zeros(observed.shape, dtype=np.uint8)

    display[observed] = 1
    display[confirmed] = 2

    # Downsample for browser performance.
    max_dimension = 350

    step = max(
        1,
        int(
            np.ceil(
                max(display.shape) / max_dimension
            )
        ),
    )

    display = display[::step, ::step]

    fig = go.Figure(
        data=go.Heatmap(
            z=display.T,
            colorscale=[
                [0.0, "#F8FAFC"],
                [0.2499, "#F8FAFC"],
                [0.25, "#F59E0B"],
                [0.7499, "#F59E0B"],
                [0.75, "#16A34A"],
                [1.0, "#16A34A"],
            ],
            zmin=0,
            zmax=2,
            colorbar=dict(
                title="State",
                tickvals=[0, 1, 2],
                ticktext=[
                    "Empty",
                    "Observed",
                    "Confirmed",
                ],
            ),
            hovertemplate=(
                "Grid column: %{x}"
                "<br>Grid row: %{y}"
                "<br>State: %{z}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Temporal Cell State — Frame {frame_id}",
        xaxis_title="Sampled grid column",
        yaxis_title="Sampled grid row",
        template="plotly_white",
        height=500,
        yaxis=dict(scaleanchor="x"),
    )

    return fig


def create_track_trajectories(tracks):
    if tracks.empty:
        return None

    required = {
        "frame",
        "track_id",
        "centroid_x",
        "centroid_y",
    }

    missing = required - set(tracks.columns)

    if missing:
        raise ValueError(
            f"Missing tracking columns: {sorted(missing)}"
        )

    data = tracks.copy()

    data["track_id"] = data["track_id"].astype(str)

    data = data.sort_values(
        ["track_id", "frame"]
    )

    fig = px.line(
        data,
        x="centroid_x",
        y="centroid_y",
        color="track_id",
        markers=True,
        title="Observed Object Track Trajectories",
        labels={
            "centroid_x": "X (m)",
            "centroid_y": "Y (m)",
            "track_id": "Track ID",
        },
        hover_data=["frame"],
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        yaxis=dict(scaleanchor="x"),
    )

    return fig


def create_track_continuity(tracks):
    if tracks.empty:
        return None

    continuity = (
        tracks.groupby("track_id")["frame"]
        .nunique()
        .reset_index(name="observed_frames")
    )

    continuity["track_id"] = (
        continuity["track_id"].astype(str)
    )

    fig = px.bar(
        continuity,
        x="track_id",
        y="observed_frames",
        title="Observed Frames per Track",
        labels={
            "track_id": "Track ID",
            "observed_frames": "Frames observed",
        },
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
    )

    return fig, continuity


def create_speed_chart(tracks):
    if tracks.empty or "speed_mps" not in tracks.columns:
        return None

    data = tracks.copy()

    data["track_id"] = data["track_id"].astype(str)

    fig = px.line(
        data.sort_values("frame"),
        x="frame",
        y="speed_mps",
        color="track_id",
        markers=True,
        title="Estimated Object Speed",
        labels={
            "frame": "Frame",
            "speed_mps": "Estimated speed (m/s)",
            "track_id": "Track ID",
        },
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
    )

    return fig
