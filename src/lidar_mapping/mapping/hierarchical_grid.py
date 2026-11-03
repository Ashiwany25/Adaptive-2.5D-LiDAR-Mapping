
"""
Nonoverlapping hierarchical 2.5D elevation mapping.

Builds occupied leaves at 0.25, 0.5, or 1.0 meters
using point-level resolution requests.

A root cell is refined to the finest resolution
requested by any of its points.

This implementation uses integer atomic-grid indices
for stable grouping and exact point conservation.
"""

import numpy as np
import pandas as pd

from lidar_mapping.utils.config import load_config


LEAF_COLUMNS = [
    "x_min",
    "y_min",
    "resolution",
    "point_count",
    "z_min",
    "z_max",
    "z_mean",
]


def build_hierarchical_grid(
    points: np.ndarray,
    requested_resolutions: np.ndarray,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Construct occupied, nonoverlapping hierarchical
    elevation cells.

    Points outside the configured XY bounds are ignored.

    Returns:
        DataFrame with one row per occupied leaf.
    """
    if config is None:
        config = load_config()

    mapping = config["mapping"]

    x_min = float(mapping["x_min"])
    x_max = float(mapping["x_max"])
    y_min = float(mapping["y_min"])
    y_max = float(mapping["y_max"])

    atomic = float(mapping["minimum_resolution"])
    root = float(mapping["root_resolution"])

    points = np.asarray(points, dtype=np.float64)
    requests = np.asarray(
        requested_resolutions, dtype=np.float64
    )

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    if requests.shape != (len(points),):
        raise ValueError(
            "Resolution requests must have shape (N,)."
        )

    allowed = np.array(
        [
            mapping["near_resolution"],
            mapping["mid_resolution"],
            mapping["far_resolution"],
        ],
        dtype=np.float64,
    )

    if not np.isfinite(requests).all():
        raise ValueError(
            "Resolution requests must be finite."
        )

    if not np.isin(requests, allowed).all():
        raise ValueError(
            "Unsupported requested resolution."
        )

    if not np.isfinite(points[:, :3]).all():
        raise ValueError(
            "Point coordinates must be finite."
        )

    if atomic <= 0 or root <= 0:
        raise ValueError(
            "Grid resolutions must be positive."
        )

    cells_per_root = root / atomic

    if not np.isclose(
        cells_per_root, round(cells_per_root)
    ):
        raise ValueError(
            "Root must be divisible by atomic resolution."
        )

    for resolution in allowed:
        ratio = resolution / atomic

        if not np.isclose(ratio, round(ratio)):
            raise ValueError(
                "Every resolution must be an integer "
                "multiple of atomic resolution."
            )

        if not np.isclose(
            root / resolution,
            round(root / resolution)
        ):
            raise ValueError(
                "Root must be divisible by each resolution."
            )

    width_atomic = (x_max - x_min) / atomic
    height_atomic = (y_max - y_min) / atomic

    if not (
        np.isclose(width_atomic, round(width_atomic))
        and np.isclose(height_atomic, round(height_atomic))
    ):
        raise ValueError(
            "Map dimensions must align to atomic grid."
        )

    if not (
        np.isclose((x_max - x_min) / root,
                   round((x_max - x_min) / root))
        and np.isclose((y_max - y_min) / root,
                       round((y_max - y_min) / root))
    ):
        raise ValueError(
            "Map dimensions must align to root grid."
        )

    valid = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )

    xyz = points[valid, :3]
    resolutions = requests[valid]

    if len(xyz) == 0:
        return pd.DataFrame(columns=LEAF_COLUMNS)

    # Atomic integer coordinates.
    ax = np.floor(
        (xyz[:, 0] - x_min) / atomic
    ).astype(np.int64)

    ay = np.floor(
        (xyz[:, 1] - y_min) / atomic
    ).astype(np.int64)

    root_units = int(round(root / atomic))

    rx = ax // root_units
    ry = ay // root_units

    data = pd.DataFrame({
        "ax": ax,
        "ay": ay,
        "rx": rx,
        "ry": ry,
        "z": xyz[:, 2],
        "requested": resolutions,
    })

    # Each root takes its finest requested resolution.
    root_requests = (
        data.groupby(["rx", "ry"])["requested"]
        .min()
        .rename("root_resolution")
        .reset_index()
    )

    data = data.merge(
        root_requests,
        on=["rx", "ry"],
        how="left",
        validate="many_to_one",
    )

    # Number of atomic cells along each leaf edge.
    leaf_units = np.rint(
        data["root_resolution"].to_numpy()
        / atomic
    ).astype(np.int64)

    # Global integer coordinates of each leaf origin.
    leaf_ax = (
        data["ax"].to_numpy() // leaf_units
    ) * leaf_units

    leaf_ay = (
        data["ay"].to_numpy() // leaf_units
    ) * leaf_units

    data["leaf_ax"] = leaf_ax
    data["leaf_ay"] = leaf_ay
    data["leaf_units"] = leaf_units

    grouped = (
        data.groupby(
            ["leaf_ax", "leaf_ay", "leaf_units"],
            sort=True,
        )
        .agg(
            point_count=("z", "size"),
            z_min=("z", "min"),
            z_max=("z", "max"),
            z_mean=("z", "mean"),
        )
        .reset_index()
    )

    result = pd.DataFrame({
        "x_min": (
            x_min + grouped["leaf_ax"] * atomic
        ),
        "y_min": (
            y_min + grouped["leaf_ay"] * atomic
        ),
        "resolution": (
            grouped["leaf_units"] * atomic
        ),
        "point_count": grouped["point_count"],
        "z_min": grouped["z_min"],
        "z_max": grouped["z_max"],
        "z_mean": grouped["z_mean"],
    })

    # Every valid input point must belong to one leaf.
    if int(result["point_count"].sum()) != len(xyz):
        raise AssertionError(
            "Point conservation check failed."
        )

    return result[LEAF_COLUMNS].reset_index(drop=True)


def summarize_hierarchical_grid(
    grid: pd.DataFrame,
) -> dict:
    """
    Summarize occupied leaves and represented points.
    """
    return {
        "occupied_leaves": int(len(grid)),
        "mapped_points": int(grid["point_count"].sum()),
        "fine_leaves": int(
            (grid["resolution"] == 0.25).sum()
        ),
        "medium_leaves": int(
            (grid["resolution"] == 0.5).sum()
        ),
        "coarse_leaves": int(
            (grid["resolution"] == 1.0).sum()
        ),
    }


def validate_nonoverlapping_grid(
    grid: pd.DataFrame,
    config: dict | None = None,
) -> bool:
    """
    Verify that occupied leaves do not overlap.

    Expands leaves into atomic-cell coordinates.
    Suitable for prototype validation and unit tests.
    """
    if config is None:
        config = load_config()

    mapping = config["mapping"]

    atomic = float(mapping["minimum_resolution"])
    x_min = float(mapping["x_min"])
    y_min = float(mapping["y_min"])

    occupied = set()

    for row in grid.itertuples(index=False):
        ax = int(round((row.x_min - x_min) / atomic))
        ay = int(round((row.y_min - y_min) / atomic))

        units = int(round(row.resolution / atomic))

        for dx in range(units):
            for dy in range(units):
                key = (ax + dx, ay + dy)

                if key in occupied:
                    return False

                occupied.add(key)

    return True
