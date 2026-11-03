
"""
Integrated 2.5D LiDAR perception pipeline.

Combines:
- Semantic layer separation
- Distance-based adaptive resolution
- Semantic override and confidence gating
- Root-cell resolution hysteresis
- Nonoverlapping hierarchical elevation mapping
- K=3 static observation persistence
- Annotated object instance extraction
- Multi-object tracking

Input points must already be pose-aligned into a
consistent reference coordinate system.

Semantic and instance labels are assumed to be
provided by the dataset.
"""

import numpy as np

from lidar_mapping.utils.config import load_config
from lidar_mapping.semantic.dynamic_separation import (
    separate_semantic_layers,
)
from lidar_mapping.semantic.resolution_override import (
    apply_semantic_override,
)
from lidar_mapping.mapping.hierarchical_grid import (
    build_hierarchical_grid,
)
from lidar_mapping.temporal.resolution_hysteresis import (
    ResolutionHysteresis,
)
from lidar_mapping.temporal.occupancy_persistence import (
    OccupancyPersistence,
    create_observation_grid,
)
from lidar_mapping.tracking.object_instances import (
    extract_object_instances,
)
from lidar_mapping.tracking.multi_object_tracker import (
    MultiObjectTracker,
)


class LidarPerceptionPipeline:
    def __init__(self, config=None):
        self.config = (
            load_config() if config is None else config
        )

        mapping = self.config["mapping"]

        self.atomic_resolution = float(
            mapping["minimum_resolution"]
        )
        self.root_resolution = float(
            mapping["root_resolution"]
        )

        self.x_min = float(mapping["x_min"])
        self.x_max = float(mapping["x_max"])
        self.y_min = float(mapping["y_min"])
        self.y_max = float(mapping["y_max"])

        nx = int(round(
            (self.x_max - self.x_min)
            / self.atomic_resolution
        ))
        ny = int(round(
            (self.y_max - self.y_min)
            / self.atomic_resolution
        ))

        self.resolution_controller = (
            ResolutionHysteresis(
                coarsening_frames=self.config[
                    "temporal"
                ]["resolution_coarsening_frames"]
            )
        )

        self.occupancy_controller = (
            OccupancyPersistence(
                grid_shape=(ny, nx),
                persistence_frames=self.config[
                    "temporal"
                ]["occupancy_persistence_frames"],
            )
        )

        tracking = self.config["tracking"]

        self.object_tracker = MultiObjectTracker(
            max_matching_distance=tracking[
                "maximum_matching_distance"
            ],
            max_missing_frames=tracking[
                "maximum_missing_frames"
            ],
            frame_interval=tracking[
                "assumed_frame_interval_seconds"
            ],
        )

        self.frame_index = -1

    def _stabilize_requests(
        self,
        points,
        requested_resolutions,
    ):
        """
        Convert point requests into root-cell requests,
        apply hysteresis, then assign stabilized root
        resolutions back to the points.
        """

        points = np.asarray(points)
        requests = np.asarray(
            requested_resolutions,
            dtype=np.float64,
        )

        if len(points) == 0:
            return requests.copy()

        valid = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
        )

        stabilized = requests.copy()

        root_x = np.floor(
            (points[valid, 0] - self.x_min)
            / self.root_resolution
        ).astype(np.int64)

        root_y = np.floor(
            (points[valid, 1] - self.y_min)
            / self.root_resolution
        ).astype(np.int64)

        valid_indices = np.flatnonzero(valid)

        root_groups = {}

        for index, rx, ry in zip(
            valid_indices, root_x, root_y
        ):
            key = (int(rx), int(ry))
            root_groups.setdefault(key, []).append(
                int(index)
            )

        for key, indices in root_groups.items():
            requested = float(
                np.min(requests[indices])
            )

            applied = self.resolution_controller.update(
                key, requested
            )

            stabilized[indices] = applied

        return stabilized

    def process_frame(
        self,
        points,
        semantic_labels,
        instance_labels,
        sensor_position=None,
        confidence=None,
    ):
        """
        Process one pose-aligned LiDAR frame.

        Confidence is optional:
        - None: semantic override uses labels ungated.
        - Array: applies configured confidence threshold.

        Returns a dictionary of perception outputs.
        """

        points = np.asarray(points)
        semantic_labels = np.asarray(semantic_labels)
        instance_labels = np.asarray(instance_labels)

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                "Points must have shape (N, 3) or (N, 4+)."
            )

        if not np.isfinite(points[:, :3]).all():
            raise ValueError(
                "Point coordinates must be finite."
            )

        if semantic_labels.shape != (len(points),):
            raise ValueError(
                "Semantic labels must have shape (N,)."
            )

        if instance_labels.shape != (len(points),):
            raise ValueError(
                "Instance labels must have shape (N,)."
            )

        if sensor_position is None:
            sensor_position = np.zeros(3)

        sensor_position = np.asarray(
            sensor_position,
            dtype=np.float64,
        )

        if (
            sensor_position.shape != (3,)
            or not np.isfinite(sensor_position).all()
        ):
            raise ValueError(
                "Sensor position must be a finite XYZ vector."
            )

        # 1. Separate static and object points.
        layers = separate_semantic_layers(
            points,
            semantic_labels,
            instance_labels,
        )

        # 2. Compute adaptive requests for all points.
        decisions = apply_semantic_override(
            points,
            semantic_labels,
            confidence=confidence,
            sensor_position=sensor_position,
            config=self.config,
        )

        # 3. Apply temporal resolution hysteresis.
        stabilized = self._stabilize_requests(
            points,
            decisions["final_resolution"],
        )

        # 4. Build independent static/object maps.
        static_map = build_hierarchical_grid(
            points[layers["static_mask"]],
            stabilized[layers["static_mask"]],
            config=self.config,
        )

        object_map = build_hierarchical_grid(
            points[layers["object_mask"]],
            stabilized[layers["object_mask"]],
            config=self.config,
        )

        # 5. K=3 persistence on static observations.
        static_observed = create_observation_grid(
            layers["static_points"],
            resolution=self.atomic_resolution,
            config=self.config,
        )

        confirmed = self.occupancy_controller.update(
            static_observed
        )

        # 6. Extract annotated object instances.
        objects = extract_object_instances(
            points,
            semantic_labels,
            instance_labels,
            frame_id=self.frame_index + 1,
            min_points=self.config[
                "tracking"
            ]["minimum_instance_points"],
        )

        # 7. Update motion-aware tracker.
        tracks = self.object_tracker.update(objects)

        self.frame_index += 1

        return {
            "frame_index": self.frame_index,
            "static_map": static_map,
            "object_map": object_map,
            "static_observed": static_observed,
            "static_confirmed": confirmed,
            "resolution_requests": (
                decisions["final_resolution"]
            ),
            "stabilized_resolutions": stabilized,
            "objects": objects,
            "tracks": tracks,
            "static_point_count": int(
                layers["static_mask"].sum()
            ),
            "object_point_count": int(
                layers["object_mask"].sum()
            ),
            "unknown_point_count": int(
                layers["unknown_mask"].sum()
            ),
        }

    def reset(self):
        """
        Clear all temporal state.
        """
        self.resolution_controller.reset()
        self.occupancy_controller.reset()
        self.object_tracker.reset()
        self.frame_index = -1
