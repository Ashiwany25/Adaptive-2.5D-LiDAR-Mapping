
"""
Temporal hysteresis for adaptive LiDAR resolution.

Rules:
- Immediately accept finer resolution.
- Delay coarsening until K consecutive requests.
- Maintain independent state for each root grid cell.

This module operates on root-cell resolution requests.
"""

from collections.abc import Mapping
from typing import Hashable

import numpy as np

from lidar_mapping.utils.config import load_config


class ResolutionHysteresis:
    """
    Stateful temporal resolution controller.

    Each root cell has:
      - Applied resolution
      - Consecutive coarsening-request counter

    Missing cells retain their existing state until
    explicitly removed or the controller is reset.
    """

    def __init__(
        self,
        coarsening_frames: int | None = None,
    ):
        if coarsening_frames is None:
            config = load_config()
            coarsening_frames = config[
                "temporal"
            ]["resolution_coarsening_frames"]

        if (
            isinstance(coarsening_frames, (bool, np.bool_))
            or not isinstance(
                coarsening_frames, (int, np.integer)
            )
            or coarsening_frames < 1
        ):
            raise ValueError(
                "Coarsening frames must be a positive integer."
            )

        self.coarsening_frames = int(coarsening_frames)

        self.states: dict[Hashable, float] = {}
        self.coarsen_counts: dict[Hashable, int] = {}

    def update(
        self,
        cell_key: Hashable,
        requested_resolution: float,
    ) -> float:
        """
        Update one root cell and return applied resolution.

        A smaller numeric resolution means finer detail.
        """

        requested = float(requested_resolution)

        if not np.isfinite(requested) or requested <= 0:
            raise ValueError(
                "Requested resolution must be finite "
                "and positive."
            )

        if cell_key not in self.states:
            self.states[cell_key] = requested
            self.coarsen_counts[cell_key] = 0
            return requested

        current = self.states[cell_key]

        if requested < current:
            # Refinement happens immediately.
            self.states[cell_key] = requested
            self.coarsen_counts[cell_key] = 0

        elif requested > current:
            # Count consecutive coarser requests.
            count = self.coarsen_counts[cell_key] + 1

            if count >= self.coarsening_frames:
                self.states[cell_key] = requested
                self.coarsen_counts[cell_key] = 0
            else:
                self.coarsen_counts[cell_key] = count

        else:
            # Stable resolution resets the counter.
            self.coarsen_counts[cell_key] = 0

        return self.states[cell_key]

    def update_frame(
        self,
        root_requests: Mapping,
    ) -> dict:
        """
        Update all observed roots in a frame.

        root_requests:
            {(root_x, root_y): requested_resolution}

        Returns:
            {(root_x, root_y): applied_resolution}
        """

        return {
            key: self.update(key, resolution)
            for key, resolution in root_requests.items()
        }

    def remove_cell(self, cell_key: Hashable) -> None:
        """
        Remove state for a root cell that is no longer
        needed.
        """
        self.states.pop(cell_key, None)
        self.coarsen_counts.pop(cell_key, None)

    def reset(self) -> None:
        """
        Reset all temporal states.
        """
        self.states.clear()
        self.coarsen_counts.clear()

    def get_state(self, cell_key: Hashable) -> float | None:
        """
        Return current applied resolution, if present.
        """
        return self.states.get(cell_key)

    def active_cell_count(self) -> int:
        """
        Return number of roots retaining state.
        """
        return len(self.states)
