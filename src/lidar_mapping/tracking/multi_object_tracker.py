
"""
Multi-object tracking using Hungarian assignment
and constant-velocity motion prediction.

Inputs:
    Object records from extract_object_instances().

Outputs:
    Active object tracks with IDs, centroids,
    velocities, and association status.

Assumes object centroids are expressed in a common
coordinate frame across consecutive LiDAR scans.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from lidar_mapping.utils.config import load_config


class MultiObjectTracker:
    def __init__(
        self,
        max_matching_distance=None,
        max_missing_frames=None,
        frame_interval=None,
    ):
        config = load_config()
        tracking = config["tracking"]

        if max_matching_distance is None:
            max_matching_distance = tracking[
                "maximum_matching_distance"
            ]

        if max_missing_frames is None:
            max_missing_frames = tracking[
                "maximum_missing_frames"
            ]

        if frame_interval is None:
            frame_interval = tracking[
                "assumed_frame_interval_seconds"
            ]

        self.max_matching_distance = float(
            max_matching_distance
        )
        self.max_missing_frames = max_missing_frames
        self.frame_interval = float(frame_interval)

        if (
            not np.isfinite(self.max_matching_distance)
            or self.max_matching_distance <= 0
        ):
            raise ValueError(
                "Matching distance must be positive."
            )

        if (
            isinstance(self.max_missing_frames, bool)
            or not isinstance(
                self.max_missing_frames,
                (int, np.integer),
            )
            or self.max_missing_frames < 0
        ):
            raise ValueError(
                "Maximum missing frames must be "
                "a nonnegative integer."
            )

        if (
            not np.isfinite(self.frame_interval)
            or self.frame_interval <= 0
        ):
            raise ValueError(
                "Frame interval must be positive."
            )

        self.max_missing_frames = int(
            self.max_missing_frames
        )

        self.tracks = {}
        self.next_track_id = 1
        self.frame_index = -1

    def _create_track(self, detection):
        track_id = self.next_track_id
        self.next_track_id += 1

        centroid = np.asarray(
            detection["centroid"],
            dtype=np.float64,
        ).copy()

        self.tracks[track_id] = {
            "track_id": track_id,
            "centroid": centroid,
            "velocity": np.zeros(3),
            "base_semantic_id": int(
                detection["base_semantic_id"]
            ),
            "instance_id": int(
                detection["instance_id"]
            ),
            "missing_frames": 0,
            "last_seen_frame": self.frame_index,
            "association": "new",
        }

    def _predict_position(self, track):
        elapsed_frames = (
            self.frame_index
            - track["last_seen_frame"]
        )

        elapsed_seconds = (
            elapsed_frames * self.frame_interval
        )

        return (
            track["centroid"]
            + track["velocity"] * elapsed_seconds
        )

    def update(self, detections):
        """
        Process one frame of object detections.

        Returns a list of tracks observed in the
        current frame. Missing tracks remain internally
        active until their tolerance expires.
        """
        self.frame_index += 1

        validated = []

        for detection in detections:
            centroid = np.asarray(
                detection["centroid"],
                dtype=np.float64,
            )

            if (
                centroid.shape != (3,)
                or not np.isfinite(centroid).all()
            ):
                raise ValueError(
                    "Each detection must have a finite "
                    "3D centroid."
                )

            validated.append({
                "centroid": centroid,
                "base_semantic_id": int(
                    detection["base_semantic_id"]
                ),
                "instance_id": int(
                    detection["instance_id"]
                ),
            })

        track_ids = list(self.tracks.keys())

        matched_tracks = set()
        matched_detections = set()

        if track_ids and validated:
            cost = np.full(
                (len(track_ids), len(validated)),
                np.inf,
                dtype=np.float64,
            )

            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                predicted = self._predict_position(
                    track
                )

                for j, detection in enumerate(validated):
                    if (
                        track["base_semantic_id"]
                        != detection["base_semantic_id"]
                    ):
                        continue

                    distance = np.linalg.norm(
                        predicted - detection["centroid"]
                    )

                    if distance <= self.max_matching_distance:
                        cost[i, j] = distance

            # Use a finite penalty for impossible matches.
            penalty = self.max_matching_distance + 1.0
            assignment_cost = np.where(
                np.isfinite(cost),
                cost,
                penalty,
            )

            row_indices, col_indices = (
                linear_sum_assignment(assignment_cost)
            )

            for i, j in zip(row_indices, col_indices):
                if not np.isfinite(cost[i, j]):
                    continue

                track_id = track_ids[i]
                track = self.tracks[track_id]
                detection = validated[j]

                elapsed_frames = (
                    self.frame_index
                    - track["last_seen_frame"]
                )

                elapsed_seconds = (
                    elapsed_frames * self.frame_interval
                )

                new_velocity = (
                    detection["centroid"]
                    - track["centroid"]
                ) / elapsed_seconds

                track["velocity"] = new_velocity
                track["centroid"] = (
                    detection["centroid"].copy()
                )
                track["instance_id"] = (
                    detection["instance_id"]
                )
                track["missing_frames"] = 0
                track["last_seen_frame"] = (
                    self.frame_index
                )
                track["association"] = "matched"

                matched_tracks.add(track_id)
                matched_detections.add(j)

        # Increment missing counters for unmatched tracks.
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]["missing_frames"] += 1

        # Remove tracks that exceed missing-frame tolerance.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if (
                track["missing_frames"]
                > self.max_missing_frames
            )
        ]

        for track_id in expired:
            del self.tracks[track_id]

        # Create new tracks for unmatched detections.
        for j, detection in enumerate(validated):
            if j not in matched_detections:
                self._create_track(detection)

        # Return only tracks observed in this frame.
        observed_tracks = []

        for track in self.tracks.values():
            if track["missing_frames"] != 0:
                continue

            observed_tracks.append({
                "frame": self.frame_index,
                "track_id": track["track_id"],
                "instance_id": track["instance_id"],
                "base_semantic_id": (
                    track["base_semantic_id"]
                ),
                "centroid": track["centroid"].copy(),
                "velocity": track["velocity"].copy(),
                "speed_mps": float(
                    np.linalg.norm(track["velocity"])
                ),
                "association": track["association"],
            })

        return sorted(
            observed_tracks,
            key=lambda item: item["track_id"],
        )

    def active_track_count(self):
        return len(self.tracks)

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_index = -1
