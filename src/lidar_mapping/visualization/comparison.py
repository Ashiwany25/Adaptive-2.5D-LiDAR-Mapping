
"""
Adaptive-vs-uniform mapping comparison.

Uses measured benchmark summaries from
SemanticKITTI Sequence 00.

The benchmark excludes semantic inference,
pose estimation, tracking and data loading.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


METHOD_ORDER = [
    "Uniform 0.25 m",
    "Uniform 0.5 m",
    "Adaptive stabilized",
]


def load_comparison_data(folder):
    folder = Path(folder)

    benchmark = pd.read_csv(
        folder / "integrated_mapping_benchmark_summary.csv"
    )

    
    accuracy = pd.read_csv(
        folder / "elevation_accuracy_summary.csv"
    )

    # Standardize column names for the dashboard.
    accuracy = accuracy.rename(
        columns={
            "Average RMSE (m)": "RMSE (m)",
            "Average MAE (m)": "MAE (m)",
        }
    )


    required_benchmark = {
        "Method",
        "Average build time (ms)",
        "Average occupied cells",
        "Average payload (KB)",
    }

    required_accuracy = {
        "Method",
        "RMSE (m)",
        "MAE (m)",
    }

    missing_benchmark = (
        required_benchmark - set(benchmark.columns)
    )

    missing_accuracy = (
        required_accuracy - set(accuracy.columns)
    )

    if missing_benchmark:
        raise ValueError(
            "Missing benchmark columns: "
            f"{sorted(missing_benchmark)}"
        )

    if missing_accuracy:
        raise ValueError(
            "Missing accuracy columns: "
            f"{sorted(missing_accuracy)}"
        )

    if benchmark["Method"].duplicated().any():
        raise ValueError(
            "Duplicate methods in benchmark summary."
        )

    if accuracy["Method"].duplicated().any():
        raise ValueError(
            "Duplicate methods in accuracy summary."
        )

    
    # Keep only the accuracy metrics needed for comparison.
    # This avoids duplicate "Average occupied cells" columns.
    accuracy = accuracy[
        ["Method", "RMSE (m)", "MAE (m)"]
    ]

    merged = benchmark.merge(
        accuracy,
        on="Method",
        how="inner",
        validate="one_to_one",
    )


    if set(merged["Method"]) != set(METHOD_ORDER):
        raise ValueError(
            "Expected all three mapping methods."
        )

    numeric_columns = [
        "Average build time (ms)",
        "Average occupied cells",
        "Average payload (KB)",
        "RMSE (m)",
        "MAE (m)",
    ]

    for column in numeric_columns:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="raise",
        )

        if not np.isfinite(
            merged[column].to_numpy()
        ).all():
            raise ValueError(
                f"Nonfinite values in {column}"
            )

        if (merged[column] < 0).any():
            raise ValueError(
                f"Negative values in {column}"
            )

    return merged.set_index("Method").loc[
        METHOD_ORDER
    ].reset_index()


def create_comparison_chart(data, metric, title):
    fig = px.bar(
        data,
        x="Method",
        y=metric,
        color="Method",
        title=title,
        category_orders={
            "Method": METHOD_ORDER
        },
        text_auto=".3s",
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=420,
        xaxis_title="Mapping method",
        yaxis_title=metric,
    )

    return fig


def calculate_relative_comparison(data):
    uniform = data.loc[
        data["Method"] == "Uniform 0.25 m"
    ].iloc[0]

    adaptive = data.loc[
        data["Method"] == "Adaptive stabilized"
    ].iloc[0]

    def percentage_change(column):
        baseline = float(uniform[column])
        proposed = float(adaptive[column])

        if baseline == 0:
            return np.nan

        return (
            (proposed - baseline) / baseline
        ) * 100.0

    return {
        "occupied_cells_change_pct": percentage_change(
            "Average occupied cells"
        ),
        "build_time_change_pct": percentage_change(
            "Average build time (ms)"
        ),
        "payload_change_pct": percentage_change(
            "Average payload (KB)"
        ),
        "rmse_change_pct": percentage_change(
            "RMSE (m)"
        ),
    }
