
"""
Multi-frame LiDAR observation persistence.

A cell becomes confirmed after K consecutive frames
with at least one point return.

This module operates on a fixed-resolution Boolean
observation grid, not on changing hierarchical leaves.

Missing returns are not interpreted as free space.
"""

import numpy as np

from lidar_mapping.utils.config import load_config


class OccupancyPersistence:
    """
    Maintain consecutive observation counters.

    K=3 by default.

    This basic model immediately clears confirmation
    when a cell is not observed in the current frame.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int],
        persistence_frames: int | None = None,
    ):
        if persistence_frames is None:
            config = load_config()
            persistence_frames = config[
                "temporal"
            ]["occupancy_persistence_frames"]

        if (
            len(grid_shape) != 2
            or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value <= 0
                for value in grid_shape
            )
        ):
            raise ValueError(
                "Grid shape must contain two positive integers."
            )

        if (
            not isinstance(persistence_frames, (int, np.integer))
            or isinstance(persistence_frames, (bool, np.bool_))
            or persistence_frames < 1
        ):
            raise ValueError(
                "Persistence frames must be a positive integer."
            )

        self.grid_shape = tuple(
            int(value) for value in grid_shape
        )
        self.persistence_frames = int(persistence_frames)

        self.counters = np.zeros(
            self.grid_shape,
            dtype=np.int32,
        )

        self.confirmed = np.zeros(
            self.grid_shape,
            dtype=bool,
        )

        self.frame_count = 0

    def update(
        self,
        observed: np.ndarray,
    ) -> np.ndarray:
        """
        Update the persistence state using one frame.

        Args:
            observed:
                Boolean grid with True wherever a
                LiDAR return was observed.

        Returns:
            Copy of the confirmed observation grid.
        """

        observed = np.asarray(observed)

        if observed.shape != self.grid_shape:
            raise ValueError(
                f"Expected shape {self.grid_shape}, "
                f"received {observed.shape}."
            )

        if observed.dtype != np.bool_:
            raise ValueError(
                "Observation grid must contain booleans."
            )

        # Increment consecutive observations, capped at K.
        self.counters[observed] = np.minimum(
            self.counters[observed] + 1,
            self.persistence_frames,
        )

        # Missing observations break the consecutive streak.
        self.counters[~observed] = 0

        self.confirmed = (
            self.counters >= self.persistence_frames
        )

        self.frame_count += 1

        return self.confirmed.copy()

    def get_pending(self) -> np.ndarray:
        """
        Cells observed recently but not yet confirmed.
        """
        return (
            (self.counters > 0)
            & ~self.confirmed
        ).copy()

    def get_counters(self) -> np.ndarray:
        """
        Return a copy of observation counters.
        """
        return self.counters.copy()

    def reset(self) -> None:
        """
        Reset all temporal observations.
        """
        self.counters.fill(0)
        self.confirmed.fill(False)
        self.frame_count = 0


def create_observation_grid(
    points: np.ndarray,
    resolution: float | None = None,
    config: dict | None = None,
) -> np.ndarray:
    """
    Convert LiDAR XYZ points into a fixed-resolution
    XY observation grid.

    Returns:
        Boolean array of shape (NY, NX).

    A True value means at least one point return
    occurred in that cell.
    """

    if config is None:
        config = load_config()

    mapping = config["mapping"]

    if resolution is None:
        resolution = float(
            mapping["minimum_resolution"]
        )

    resolution = float(resolution)

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError(
            "Resolution must be finite and positive."
        )

    x_min = float(mapping["x_min"])
    x_max = float(mapping["x_max"])
    y_min = float(mapping["y_min"])
    y_max = float(mapping["y_max"])

    nx_float = (x_max - x_min) / resolution
    ny_float = (y_max - y_min) / resolution

    if not (
        np.isclose(nx_float, round(nx_float))
        and np.isclose(ny_float, round(ny_float))
    ):
        raise ValueError(
            "Map dimensions must be divisible by resolution."
        )

    nx = int(round(nx_float))
    ny = int(round(ny_float))

    grid = np.zeros((ny, nx), dtype=bool)

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4+)."
        )

    valid = (
        np.isfinite(points[:, :3]).all(axis=1)
        & (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
    )

    xyz = points[valid, :3]

    if len(xyz) == 0:
        return grid

    ix = np.floor(
        (xyz[:, 0] - x_min) / resolution
    ).astype(np.int64)

    iy = np.floor(
        (xyz[:, 1] - y_min) / resolution
    ).astype(np.int64)

    # Protect against floating-point edge effects.
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    grid[iy, ix] = True

    return grid
