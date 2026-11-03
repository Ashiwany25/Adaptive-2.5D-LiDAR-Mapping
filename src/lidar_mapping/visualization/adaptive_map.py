
"""
Interactive adaptive-resolution 2.5D map visualization.

Colors represent occupied hierarchical cell resolution,
not mapping accuracy or safety.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


RESOLUTION_COLORS = {
    0.25: "#16A34A",
    0.50: "#EAB308",
    1.00: "#EF4444",
}


def create_adaptive_resolution_map(
    static_map,
    object_map,
    tracks=None,
    show_static=True,
    show_objects=True,
    show_tracks=True,
    frame_id=None,
):
    fig = go.Figure()

    def add_cells(df, layer_name, opacity):
        if df is None or df.empty:
            return

        required = {
            "x_min", "y_min", "resolution",
            "point_count", "z_min", "z_max", "z_mean"
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing map columns: {sorted(missing)}"
            )

        for resolution, color in RESOLUTION_COLORS.items():
            subset = df[
                np.isclose(
                    df["resolution"].to_numpy(
                        dtype=float
                    ),
                    resolution,
                )
            ]

            if subset.empty:
                continue

            x_values = []
            y_values = []

            for row in subset.itertuples(index=False):
                x = float(row.x_min)
                y = float(row.y_min)

                x_values.extend([
                    x, x + resolution,
                    x + resolution, x, x, None
                ])

                y_values.extend([
                    y, y, y + resolution,
                    y + resolution, y, None
                ])

            fig.add_trace(
                go.Scattergl(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=1.4 if layer_name == "Objects" else 0.8,
                    ),
                    opacity=opacity,
                    name=f"{layer_name} | {resolution:.2f} m",
                    hoverinfo="skip",
                )
            )

            centers_x = (
                subset["x_min"] + resolution / 2
            )

            centers_y = (
                subset["y_min"] + resolution / 2
            )

            custom = np.column_stack([
                subset["point_count"],
                subset["z_min"],
                subset["z_max"],
                subset["z_mean"],
            ])

            fig.add_trace(
                go.Scattergl(
                    x=centers_x,
                    y=centers_y,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=color,
                        opacity=0.4,
                    ),
                    customdata=custom,
                    hovertemplate=(
                        f"{layer_name}"
                        f"<br>Resolution: {resolution:.2f} m"
                        "<br>X: %{x:.2f} m"
                        "<br>Y: %{y:.2f} m"
                        "<br>Points: %{customdata[0]:.0f}"
                        "<br>Min elevation: %{customdata[1]:.2f} m"
                        "<br>Max elevation: %{customdata[2]:.2f} m"
                        "<br>Mean elevation: %{customdata[3]:.2f} m"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    if show_static:
        add_cells(static_map, "Static", opacity=0.65)

    if show_objects:
        add_cells(object_map, "Objects", opacity=0.95)

    if (
        show_tracks
        and tracks is not None
        and not tracks.empty
    ):
        fig.add_trace(
            go.Scatter(
                x=tracks["centroid_x"],
                y=tracks["centroid_y"],
                mode="markers+text",
                text=[
                    f"ID {int(track_id)}"
                    for track_id in tracks["track_id"]
                ],
                textposition="top center",
                marker=dict(
                    size=10,
                    color="#2563EB",
                    symbol="diamond",
                ),
                name="Tracked objects",
                hovertemplate=(
                    "Track %{text}"
                    "<br>X: %{x:.2f} m"
                    "<br>Y: %{y:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    title = "Adaptive Resolution 2.5D LiDAR Map"

    if frame_id is not None:
        title += f" — Frame {frame_id}"

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=50, r=30, t=90, b=50),
    )

    return fig


def summarize_resolution_cells(static_map, object_map):
    rows = []

    for name, df in [
        ("Static", static_map),
        ("Objects", object_map),
    ]:
        for resolution in [0.25, 0.50, 1.00]:
            count = int(
                np.isclose(
                    df["resolution"].to_numpy(
                        dtype=float
                    ),
                    resolution,
                ).sum()
            )

            rows.append({
                "Layer": name,
                "Resolution (m)": resolution,
                "Occupied cells": count,
            })

    return pd.DataFrame(rows)
