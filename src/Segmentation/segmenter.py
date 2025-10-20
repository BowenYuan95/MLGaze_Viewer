"""Main task-based segmentation implementation with sliding window approach."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .data_types import TaskSegmentResult, SegmentationConfig
# Signal processing removed - only keeping gaze deviation calculation inline


def calculate_gaze_deviation_degrees(gaze_direction_1: np.ndarray,
                                   gaze_direction_2: np.ndarray) -> float:
    """Calculate angle between two gaze direction vectors in degrees.

    Args:
        gaze_direction_1: First normalized 3D direction vector [x, y, z]
        gaze_direction_2: Second normalized 3D direction vector [x, y, z]

    Returns:
        Angle in degrees between the two gaze directions
    """
    # Ensure vectors are normalized
    norm1 = np.linalg.norm(gaze_direction_1)
    norm2 = np.linalg.norm(gaze_direction_2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # Handle zero vectors

    dir1_normalized = gaze_direction_1 / norm1
    dir2_normalized = gaze_direction_2 / norm2

    # Calculate dot product and clamp to handle numerical errors
    dot_product = np.clip(np.dot(dir1_normalized, dir2_normalized), -1.0, 1.0)

    # Calculate angle in radians then convert to degrees
    angle_radians = np.arccos(dot_product)
    return np.degrees(angle_radians)


class TaskSegmenter:
    """Main class for performing task-based session segmentation using sliding window approach."""

    def __init__(self, config: SegmentationConfig):
        """Initialize the task segmenter.

        Args:
            config: Configuration parameters for segmentation
        """
        self.config = config

    def segment(self, session_data) -> List[TaskSegmentResult]:
        """Perform task-based segmentation on session data.

        Args:
            session_data: SessionData containing camera, gaze, and optional hand tracking data

        Returns:
            List of detected task segments
        """
        # Step 1: Extract and validate data
        if not self._validate_session_data(session_data):
            return []

        # Step 2: Extract primary camera positions and timestamps
        positions, timestamps = self._extract_camera_positions(session_data)
        if len(positions) == 0:
            return []

        # Step 3: Extract gaze data (no preprocessing)
        gaze_data = self._process_gaze_data(session_data)

        # Step 4: Extract hand tracking data (if enabled)
        hand_tracking_data = None
        if self.config.require_hand_tracking:
            hand_tracking_data = self._process_hand_tracking_data(session_data)

        # Step 5: Apply sliding window stability detection
        stable_timestamps = self._apply_sliding_window(
            positions, timestamps, gaze_data, hand_tracking_data, session_data
        )

        # Step 6: Create candidate segments from stable periods
        candidate_segments = self._create_candidate_segments(
            stable_timestamps, positions, timestamps
        )

        # Step 7: Apply time buffers
        buffered_segments = self._apply_time_buffers(candidate_segments, session_data)

        # Step 8: Merge consecutive segments
        merged_segments = self._merge_segments(buffered_segments)

        # Step 9: Consolidate locations
        consolidated_segments = self._consolidate_locations(merged_segments)

        # Step 10: Refine boundaries using gaze transitions (optional)
        if self.config.enable_gaze_boundary_refinement and gaze_data is not None:
            final_segments = self._refine_segment_boundaries(
                consolidated_segments, gaze_data, positions, timestamps
            )
        else:
            final_segments = consolidated_segments

        # Step 11: Generate results with proper IDs
        return self._generate_results(final_segments)

    def _validate_session_data(self, session_data) -> bool:
        """Validate that session data contains required information."""
        if not hasattr(session_data, 'camera_metadata') or not session_data.camera_metadata:
            return False

        if not hasattr(session_data, 'primary_camera') or not session_data.primary_camera:
            return False

        if session_data.primary_camera not in session_data.camera_metadata:
            return False

        return True

    def _extract_camera_positions(self, session_data) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D positions and timestamps from primary camera metadata."""
        primary_metadata = session_data.camera_metadata[session_data.primary_camera]

        if 'posX' not in primary_metadata.columns:
            return np.array([]), np.array([])

        # Extract positions and timestamps
        positions = primary_metadata[['posX', 'posY', 'posZ']].values
        timestamps = primary_metadata['timestamp'].values

        # Sort by timestamp
        sort_indices = np.argsort(timestamps)
        positions = positions[sort_indices]
        timestamps = timestamps[sort_indices]

        return positions, timestamps


    def _process_gaze_data(self, session_data) -> Optional[pd.DataFrame]:
        """Extract gaze data and filter out unwanted gaze states.

        Filters out gaze states specified in config.filtered_gaze_states
        (default: ["Blink", "Unknown"])
        """
        if not self.config.require_gaze_stability:
            return None

        if not hasattr(session_data, 'gaze') or session_data.gaze.empty:
            return None

        gaze_df = session_data.gaze.copy()

        # Filter out unwanted gaze states (Blink, Unknown, etc.)
        if 'gazeState' in gaze_df.columns and self.config.filtered_gaze_states:
            gaze_df = gaze_df[~gaze_df['gazeState'].isin(self.config.filtered_gaze_states)]

        return gaze_df

    def _process_hand_tracking_data(self, session_data) -> Optional[pd.DataFrame]:
        """Process hand tracking data if available."""
        if not hasattr(session_data, 'hand_tracking') or session_data.hand_tracking is None:
            return None

        return session_data.hand_tracking.copy()

    def _apply_sliding_window(self,
                            positions: np.ndarray,
                            timestamps: np.ndarray,
                            gaze_data: Optional[pd.DataFrame],
                            hand_tracking_data: Optional[pd.DataFrame],
                            session_data) -> List[int]:
        """Apply sliding window stability detection with mode switching.

        Supports two modes:
        - "strict": All samples must be stable (original behavior)
        - "percentage": Allow configurable percentage of unstable samples

        Falls back to strict mode if window has fewer than min_samples_for_percentage_mode.

        Returns:
            List of timestamps that are considered stable
        """
        if len(positions) == 0:
            return []

        stable_timestamps = []
        window_duration_ns = int(self.config.stability_time_window_s * 1e9)

        for i, current_timestamp in enumerate(timestamps):
            window_end = current_timestamp + window_duration_ns

            # Get all samples within the sliding window
            window_mask = (timestamps >= current_timestamp) & (timestamps <= window_end)
            window_positions = positions[window_mask]
            window_timestamps = timestamps[window_mask]

            if len(window_positions) < 2:
                continue  # Need at least 2 samples for stability check

            # Determine which mode to use
            use_percentage_mode = (
                self.config.stability_mode == "percentage" and
                len(window_positions) >= self.config.min_samples_for_percentage_mode
            )

            # Check all stability criteria within this window
            is_stable = True

            if use_percentage_mode:
                # PERCENTAGE MODE - allow tolerance for unstable samples

                # 1. Spatial stability check
                if self.config.require_spatial_stability:
                    head_unstable_pct = self._check_spatial_stability_percentage(window_positions)
                    if head_unstable_pct > self.config.head_instability_threshold_percent:
                        is_stable = False

                # 2. Gaze stability check
                if self.config.require_gaze_stability and gaze_data is not None:
                    gaze_unstable_pct = self._check_gaze_stability_percentage(
                        window_timestamps, gaze_data, session_data
                    )
                    if gaze_unstable_pct > self.config.gaze_instability_threshold_percent:
                        is_stable = False

                # 3. Hand tracking check
                if self.config.require_hand_tracking and hand_tracking_data is not None:
                    hand_untracked_pct = self._check_hand_tracking_percentage(
                        window_timestamps, hand_tracking_data
                    )
                    if hand_untracked_pct > self.config.hand_untracked_threshold_percent:
                        is_stable = False

            else:
                # STRICT MODE - all samples must be stable (original behavior)

                # 1. Spatial stability check
                if self.config.require_spatial_stability:
                    if not self._check_spatial_stability_in_window(window_positions):
                        is_stable = False

                # 2. Gaze stability check
                if self.config.require_gaze_stability and gaze_data is not None:
                    if not self._check_gaze_stability_in_window(
                        window_timestamps, gaze_data, session_data
                    ):
                        is_stable = False

                # 3. Hand tracking check
                if self.config.require_hand_tracking and hand_tracking_data is not None:
                    if not self._check_hand_tracking_in_window(
                        window_timestamps, hand_tracking_data
                    ):
                        is_stable = False

            if is_stable:
                stable_timestamps.append(current_timestamp)

        return stable_timestamps

    def _check_spatial_stability_in_window(self, window_positions: np.ndarray) -> bool:
        """Check if all positions in window are within spatial stability radius.

        Distance calculation: Only X and Z (horizontal plane), ignoring Y (vertical)
        """
        if len(window_positions) < 2:
            return False

        start_position = window_positions[0]

        for position in window_positions[1:]:
            # Calculate horizontal distance only (X-Z plane, ignore Y)
            horizontal_diff = np.array([
                position[0] - start_position[0],  # X
                position[2] - start_position[2]   # Z
            ])
            distance = np.linalg.norm(horizontal_diff)

            if distance > self.config.spatial_stability_radius_m:
                return False

        return True

    def _check_gaze_stability_in_window(self,
                                      window_timestamps: np.ndarray,
                                      gaze_data: pd.DataFrame,
                                      session_data) -> bool:
        """Check gaze stability within window."""
        # Find gaze samples within the window
        min_ts, max_ts = window_timestamps[0], window_timestamps[-1]
        window_gaze = gaze_data[
            (gaze_data['timestamp'] >= min_ts) &
            (gaze_data['timestamp'] <= max_ts)
        ]

        if len(window_gaze) < 2:
            return False

        # Check gaze deviation between consecutive samples
        if all(col in window_gaze.columns for col in ['gazeDirectionX', 'gazeDirectionY', 'gazeDirectionZ']):
            directions = window_gaze[['gazeDirectionX', 'gazeDirectionY', 'gazeDirectionZ']].values

            for i in range(len(directions) - 1):
                deviation = calculate_gaze_deviation_degrees(
                    directions[i], directions[i + 1]
                )
                if deviation > self.config.gaze_deviation_threshold_deg:
                    return False

        # Check focus distance if required
        if hasattr(self.config, 'focus_distance_threshold_m'):
            primary_metadata = session_data.camera_metadata[session_data.primary_camera]

            for _, gaze_row in window_gaze.iterrows():
                # Find corresponding camera position for this timestamp
                camera_pos = self._get_camera_position_at_timestamp(
                    gaze_row['timestamp'], primary_metadata
                )

                if camera_pos is not None:
                    # Calculate focus point from gaze origin and direction
                    if all(col in gaze_row for col in ['gazeOriginX', 'gazeOriginY', 'gazeOriginZ']):
                        gaze_origin = np.array([
                            gaze_row['gazeOriginX'], gaze_row['gazeOriginY'], gaze_row['gazeOriginZ']
                        ])

                        # Calculate distance from head to gaze origin
                        focus_distance = np.linalg.norm(gaze_origin - camera_pos)

                        if focus_distance > self.config.focus_distance_threshold_m:
                            return False

        return True

    def _check_hand_tracking_in_window(self,
                                     window_timestamps: np.ndarray,
                                     hand_tracking_data: pd.DataFrame) -> bool:
        """Check hand tracking stability within window."""
        min_ts, max_ts = window_timestamps[0], window_timestamps[-1]

        # Get all hand tracking samples within window
        window_hands = hand_tracking_data[
            (hand_tracking_data['timestamp'] >= min_ts) &
            (hand_tracking_data['timestamp'] <= max_ts)
        ]

        if window_hands.empty:
            return False  # No hand tracking data in window

        # Group by timestamp and check if at least one hand is tracked at each timestamp
        for timestamp in window_hands['timestamp'].unique():
            timestamp_hands = window_hands[window_hands['timestamp'] == timestamp]

            # Check if at least one hand has isTracked = True
            if 'isTracked' in timestamp_hands.columns:
                tracked_hands = timestamp_hands[timestamp_hands['isTracked'] == True]
                if len(tracked_hands) == 0:
                    return False  # No hands tracked at this timestamp

        return True

    def _check_spatial_stability_percentage(self, window_positions: np.ndarray) -> float:
        """Calculate percentage of unstable head positions in window.

        Reference: FIRST position in window (matches current strict mode)
        Distance calculation: Only X and Z (horizontal plane), ignoring Y (vertical)

        Args:
            window_positions: Array of 3D positions in window

        Returns:
            Percentage of positions outside stability radius (0-100)
        """
        if len(window_positions) < 2:
            return 100.0  # Not enough data = 100% unstable

        reference_position = window_positions[0]  # FIRST position
        unstable_count = 0

        for position in window_positions[1:]:
            # Calculate horizontal distance only (X-Z plane, ignore Y)
            horizontal_diff = np.array([
                position[0] - reference_position[0],  # X
                position[2] - reference_position[2]   # Z
            ])
            distance = np.linalg.norm(horizontal_diff)

            if distance > self.config.spatial_stability_radius_m:
                unstable_count += 1

        unstable_percentage = (unstable_count / (len(window_positions) - 1)) * 100.0
        return unstable_percentage

    def _check_gaze_stability_percentage(self,
                                        window_timestamps: np.ndarray,
                                        gaze_data: pd.DataFrame,
                                        session_data) -> float:
        """Calculate percentage of unstable gaze samples in window.

        Reference: CONSECUTIVE sample pairs (matches current strict mode)
        Includes both deviation violations AND focus distance violations

        Args:
            window_timestamps: Timestamps in window
            gaze_data: Filtered gaze DataFrame (Blink/Unknown already removed)
            session_data: Session data for camera positions

        Returns:
            Percentage of unstable gaze samples (0-100)
        """
        min_ts, max_ts = window_timestamps[0], window_timestamps[-1]
        window_gaze = gaze_data[
            (gaze_data['timestamp'] >= min_ts) &
            (gaze_data['timestamp'] <= max_ts)
        ]

        if len(window_gaze) < 2:
            return 100.0  # Not enough data = 100% unstable

        if not all(col in window_gaze.columns for col in ['gazeDirectionX', 'gazeDirectionY', 'gazeDirectionZ']):
            return 100.0

        unstable_count = 0
        total_checks = 0

        # Check gaze deviation between CONSECUTIVE samples
        directions = window_gaze[['gazeDirectionX', 'gazeDirectionY', 'gazeDirectionZ']].values
        for i in range(len(directions) - 1):
            deviation = calculate_gaze_deviation_degrees(
                directions[i], directions[i + 1]
            )
            if deviation > self.config.gaze_deviation_threshold_deg:
                unstable_count += 1
            total_checks += 1

        # Check focus distance if required
        if hasattr(self.config, 'focus_distance_threshold_m'):
            primary_metadata = session_data.camera_metadata[session_data.primary_camera]

            for _, gaze_row in window_gaze.iterrows():
                camera_pos = self._get_camera_position_at_timestamp(
                    gaze_row['timestamp'], primary_metadata
                )

                if camera_pos is not None:
                    if all(col in gaze_row for col in ['gazeOriginX', 'gazeOriginY', 'gazeOriginZ']):
                        gaze_origin = np.array([
                            gaze_row['gazeOriginX'], gaze_row['gazeOriginY'], gaze_row['gazeOriginZ']
                        ])

                        focus_distance = np.linalg.norm(gaze_origin - camera_pos)

                        if focus_distance > self.config.focus_distance_threshold_m:
                            unstable_count += 1
                        total_checks += 1

        if total_checks == 0:
            return 100.0

        unstable_percentage = (unstable_count / total_checks) * 100.0
        return unstable_percentage

    def _check_hand_tracking_percentage(self,
                                       window_timestamps: np.ndarray,
                                       hand_tracking_data: pd.DataFrame) -> float:
        """Calculate percentage of time with NO hand tracking in window.

        Args:
            window_timestamps: Timestamps in window
            hand_tracking_data: Hand tracking DataFrame

        Returns:
            Percentage of timestamps with no tracked hands (0-100)
        """
        min_ts, max_ts = window_timestamps[0], window_timestamps[-1]

        window_hands = hand_tracking_data[
            (hand_tracking_data['timestamp'] >= min_ts) &
            (hand_tracking_data['timestamp'] <= max_ts)
        ]

        if window_hands.empty:
            return 100.0  # No data = 100% untracked

        unique_timestamps = window_hands['timestamp'].unique()
        untracked_count = 0

        for timestamp in unique_timestamps:
            timestamp_hands = window_hands[window_hands['timestamp'] == timestamp]

            if 'isTracked' in timestamp_hands.columns:
                tracked_hands = timestamp_hands[timestamp_hands['isTracked'] == True]
                if len(tracked_hands) == 0:
                    untracked_count += 1  # No hands tracked at this timestamp

        untracked_percentage = (untracked_count / len(unique_timestamps)) * 100.0
        return untracked_percentage

    def _get_camera_position_at_timestamp(self,
                                        timestamp: int,
                                        camera_metadata: pd.DataFrame) -> Optional[np.ndarray]:
        """Get camera position at specific timestamp using nearest neighbor."""
        if camera_metadata.empty:
            return None

        # Find closest timestamp
        time_diffs = np.abs(camera_metadata['timestamp'] - timestamp)
        closest_idx = np.argmin(time_diffs)

        closest_row = camera_metadata.iloc[closest_idx]

        if all(col in closest_row for col in ['posX', 'posY', 'posZ']):
            return np.array([closest_row['posX'], closest_row['posY'], closest_row['posZ']])

        return None

    def _create_candidate_segments(self,
                                 stable_timestamps: List[int],
                                 positions: np.ndarray,
                                 timestamps: np.ndarray) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        """Create candidate segments from stable timestamp periods."""
        if not stable_timestamps:
            return []

        segments = []
        stable_timestamps = sorted(stable_timestamps)

        # Group consecutive stable timestamps into segments
        current_start = stable_timestamps[0]
        current_end = stable_timestamps[0]

        for i in range(1, len(stable_timestamps)):
            timestamp = stable_timestamps[i]

            # Check if this timestamp is consecutive (within reasonable gap)
            gap_ns = timestamp - current_end
            gap_s = gap_ns / 1e9

            if gap_s <= self.config.merge_consecutive_gap_s:
                current_end = timestamp
            else:
                # End current segment and start a new one
                if current_end - current_start >= self.config.min_segment_duration_s * 1e9:
                    start_pos, end_pos = self._get_positions_for_segment(
                        current_start, current_end, positions, timestamps
                    )
                    if start_pos is not None and end_pos is not None:
                        segments.append((current_start, current_end, start_pos, end_pos))

                current_start = timestamp
                current_end = timestamp

        # Add the final segment
        if current_end - current_start >= self.config.min_segment_duration_s * 1e9:
            start_pos, end_pos = self._get_positions_for_segment(
                current_start, current_end, positions, timestamps
            )
            if start_pos is not None and end_pos is not None:
                segments.append((current_start, current_end, start_pos, end_pos))

        return segments

    def _get_positions_for_segment(self,
                                 start_ts: int,
                                 end_ts: int,
                                 positions: np.ndarray,
                                 timestamps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get start and end positions for a segment."""
        if len(positions) == 0 or len(timestamps) == 0:
            return None, None

        # Find closest positions to start and end timestamps
        start_idx = np.argmin(np.abs(timestamps - start_ts))
        end_idx = np.argmin(np.abs(timestamps - end_ts))

        start_pos = positions[start_idx]
        end_pos = positions[end_idx]

        return start_pos, end_pos

    def _apply_time_buffers(self,
                          segments: List[Tuple[int, int, np.ndarray, np.ndarray]],
                          session_data) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        """Apply configurable time buffers to segment boundaries."""
        if not segments:
            return segments

        buffered_segments = []
        start_buffer_ns = int(self.config.stability_buffer_start_s * 1e9)
        end_buffer_ns = int(self.config.stability_buffer_end_s * 1e9)

        # Get session time bounds
        session_start = session_data.start_timestamp
        session_end = session_data.end_timestamp

        for start_ts, end_ts, start_pos, end_pos in segments:
            # Apply buffers while respecting session boundaries
            buffered_start = max(session_start, start_ts - start_buffer_ns)
            buffered_end = min(session_end, end_ts + end_buffer_ns)

            buffered_segments.append((buffered_start, buffered_end, start_pos, end_pos))

        return buffered_segments

    def _merge_segments(self,
                       segments: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
        """Merge segments that overlap or touch after time buffering.

        Only merges segments where next_start <= current_end (overlapping/touching).
        Uses outer boundary positions (first segment's start_pos + last segment's end_pos).
        """
        if len(segments) <= 1:
            return segments

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            current_start, current_end, current_start_pos, current_end_pos = current_segment
            next_start, next_end, next_start_pos, next_end_pos = next_segment

            # Only merge if segments overlap or touch
            if next_start <= current_end:
                # Merge segments - use outer boundaries for positions
                current_segment = (current_start, next_end, current_start_pos, next_end_pos)
            else:
                # Keep current segment and move to next
                merged.append(current_segment)
                current_segment = next_segment

        # Add the final segment
        merged.append(current_segment)
        return merged

    def _consolidate_locations(self,
                             segments: List[Tuple[int, int, np.ndarray, np.ndarray]]) -> List[Tuple[int, int, np.ndarray, np.ndarray, bool]]:
        """Consolidate segment locations if start/end are close."""
        consolidated = []

        for start_ts, end_ts, start_pos, end_pos in segments:
            distance = np.linalg.norm(end_pos - start_pos)
            is_consolidated = distance < self.config.location_consolidation_threshold_m

            consolidated.append((start_ts, end_ts, start_pos, end_pos, is_consolidated))

        return consolidated

    def _detect_gaze_transitions(self,
                                  gaze_data: pd.DataFrame,
                                  start_ts: int,
                                  end_ts: int,
                                  boundary_type: str = "start") -> Tuple[bool, int]:
        """Detect gaze transition periods at segment boundaries.

        Args:
            gaze_data: Gaze dataframe with timestamp, direction, origin columns
            start_ts: Search window start timestamp (nanoseconds)
            end_ts: Search window end timestamp (nanoseconds)
            boundary_type: "start" or "end" - which boundary we're analyzing

        Returns:
            Tuple of (is_transition_detected, stable_timestamp)
            - is_transition_detected: True if unstable gaze detected
            - stable_timestamp: First/last stable timestamp in window
        """
        # Get gaze samples in search window
        window_gaze = gaze_data[
            (gaze_data['timestamp'] >= start_ts) &
            (gaze_data['timestamp'] <= end_ts)
        ].copy()

        if len(window_gaze) < 2:
            return False, start_ts if boundary_type == "start" else end_ts

        # Convert to nanoseconds for consistency
        bin_size_ns = int(self.config.saccade_detection_bin_s * 1e9)

        # Create time bins
        min_ts = window_gaze['timestamp'].min()
        max_ts = window_gaze['timestamp'].max()
        bins = np.arange(min_ts, max_ts + bin_size_ns, bin_size_ns)

        # Analyze each bin for instability
        bin_is_unstable = []

        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]

            bin_samples = window_gaze[
                (window_gaze['timestamp'] >= bin_start) &
                (window_gaze['timestamp'] < bin_end)
            ]

            if len(bin_samples) < 2:
                bin_is_unstable.append(False)
                continue

            # Check 1: High-frequency saccades
            saccade_count = 0
            directions = bin_samples[['gazeDirectionX', 'gazeDirectionY', 'gazeDirectionZ']].values

            for j in range(len(directions) - 1):
                deviation = calculate_gaze_deviation_degrees(directions[j], directions[j+1])
                if deviation > self.config.saccade_threshold_deg:
                    saccade_count += 1

            # Check 2: Focus distance variability
            if 'gazeOriginX' in bin_samples.columns:
                origins = bin_samples[['gazeOriginX', 'gazeOriginY', 'gazeOriginZ']].values
                focus_distances = np.linalg.norm(origins, axis=1)
                focus_std = np.std(focus_distances) if len(focus_distances) > 1 else 0.0

                # Check for absolute focus shift
                focus_shift = np.max(focus_distances) - np.min(focus_distances) if len(focus_distances) > 1 else 0.0
            else:
                focus_std = 0.0
                focus_shift = 0.0

            # Bin is unstable if either criterion is met
            is_unstable = (
                saccade_count >= self.config.saccade_rate_threshold or
                focus_std > self.config.focus_transition_threshold_m or
                focus_shift > self.config.min_focus_shift_m
            )

            bin_is_unstable.append(is_unstable)

        # Find transition boundaries
        # For START: we want to find where instability ENDS (last unstable bin before stability)
        # For END: we want to find where instability BEGINS (first unstable bin after stability)

        if boundary_type == "start":
            # Search forward from start to find last unstable bin
            last_unstable_idx = -1
            for i, unstable in enumerate(bin_is_unstable):
                if unstable:
                    last_unstable_idx = i
                else:
                    # Found stable bin after unstable period
                    if last_unstable_idx >= 0:
                        # Return end of last unstable bin (transition detected)
                        return True, bins[last_unstable_idx + 1]
                    # Stable from the start
                    return False, start_ts

            # All unstable or ended in unstable
            if last_unstable_idx >= 0:
                return True, bins[last_unstable_idx + 1]
            return False, start_ts

        else:  # end
            # Search backward from end to find first unstable bin
            first_unstable_idx = -1
            for i in range(len(bin_is_unstable) - 1, -1, -1):
                if bin_is_unstable[i]:
                    first_unstable_idx = i
                else:
                    # Found stable bin before unstable period
                    if first_unstable_idx >= 0:
                        # Return start of first unstable bin (transition detected)
                        return True, bins[first_unstable_idx]
                    # Stable until the end
                    return False, end_ts

            # All unstable or started with unstable
            if first_unstable_idx >= 0:
                return True, bins[first_unstable_idx]
            return False, end_ts

    def _refine_segment_boundaries(self,
                                   segments: List[Tuple[int, int, np.ndarray, np.ndarray, bool]],
                                   gaze_data: pd.DataFrame,
                                   positions: np.ndarray,
                                   timestamps: np.ndarray) -> List[Tuple[int, int, np.ndarray, np.ndarray, bool, Dict]]:
        """Refine segment boundaries by trimming gaze transition periods.

        Args:
            segments: List of initial segments (start_ts, end_ts, start_pos, end_pos, is_consolidated)
            gaze_data: Gaze dataframe for transition detection
            positions: Camera positions array
            timestamps: Camera timestamps array

        Returns:
            List of refined segments with refinement metadata
            (start_ts, end_ts, start_pos, end_pos, is_consolidated, metadata_dict)
        """
        if gaze_data is None or gaze_data.empty:
            # No gaze data - return segments unchanged with empty metadata
            return [(start, end, start_pos, end_pos, cons, {})
                    for start, end, start_pos, end_pos, cons in segments]

        refined_segments = []
        search_window_ns = int(self.config.transition_search_window_s * 1e9)

        for original_start, original_end, _, _, is_consolidated in segments:
            original_duration = original_end - original_start

            # Define search windows INSIDE the segment
            # We search inward from the boundaries to find where stability begins/ends
            start_search_begin = original_start
            start_search_end = min(original_start + search_window_ns, original_end)

            end_search_begin = max(original_end - search_window_ns, original_start)
            end_search_end = original_end

            # Detect transitions
            # For start: find where instability ends (trim from start to this point)
            start_transition, refined_start = self._detect_gaze_transitions(
                gaze_data, start_search_begin, start_search_end, "start"
            )

            # For end: find where instability begins (trim from this point to end)
            end_transition, refined_end = self._detect_gaze_transitions(
                gaze_data, end_search_begin, end_search_end, "end"
            )

            # Apply shrinking constraints
            new_duration = refined_end - refined_start
            shrink_amount = original_duration - new_duration
            shrink_percent = (shrink_amount / original_duration) * 100 if original_duration > 0 else 0

            # Don't shrink too much
            if shrink_percent > self.config.max_shrink_percent:
                # Proportionally reduce the trimming
                allowed_shrink = (self.config.max_shrink_percent / 100.0) * original_duration
                excess_shrink = shrink_amount - allowed_shrink

                # Distribute excess back proportionally
                start_trim = original_start - refined_start
                end_trim = refined_end - original_end
                total_trim = start_trim + abs(end_trim)

                if total_trim > 0:
                    start_reduction = int((start_trim / total_trim) * excess_shrink)
                    end_reduction = int((abs(end_trim) / total_trim) * excess_shrink)

                    refined_start = original_start - (start_trim - start_reduction)
                    refined_end = original_end + (abs(end_trim) - end_reduction)

            # Ensure minimum duration
            if (refined_end - refined_start) / 1e9 < self.config.min_stable_core_duration_s:
                # Segment too short after refinement - keep original
                refined_start = original_start
                refined_end = original_end

            # Get refined positions
            refined_start_idx = np.argmin(np.abs(timestamps - refined_start))
            refined_end_idx = np.argmin(np.abs(timestamps - refined_end))

            refined_start_pos = positions[refined_start_idx]
            refined_end_pos = positions[refined_end_idx]

            # Calculate boundary confidence (0-1, higher = more confident)
            # Based on whether transitions were detected
            confidence = 0.5  # Baseline
            if start_transition:
                confidence += 0.25
            if end_transition:
                confidence += 0.25

            # Create metadata
            metadata = {
                'original_start': original_start,
                'original_end': original_end,
                'trimmed_start_ns': refined_start - original_start,
                'trimmed_end_ns': original_end - refined_end,
                'confidence': confidence
            }

            refined_segments.append((
                refined_start, refined_end,
                refined_start_pos, refined_end_pos,
                is_consolidated, metadata
            ))

        return refined_segments

    def _generate_results(self,
                        segments: List[Tuple[int, int, np.ndarray, np.ndarray, bool]]) -> List[TaskSegmentResult]:
        """Generate final TaskSegmentResult objects."""
        results = []

        for i, segment_data in enumerate(segments):
            # Handle both old format (5-tuple) and new format (6-tuple with metadata)
            if len(segment_data) == 6:
                start_ts, end_ts, start_pos, end_pos, is_consolidated, metadata = segment_data
            else:
                start_ts, end_ts, start_pos, end_pos, is_consolidated = segment_data
                metadata = {}

            segment_id = f"seg_task_{i+1:03d}"
            duration_ns = end_ts - start_ts

            result = TaskSegmentResult(
                segmentation_id=segment_id,
                start_timestamp=start_ts,
                finish_timestamp=end_ts,
                duration_ns=duration_ns,
                start_location=start_pos,
                finish_location=end_pos,
                is_location_consolidated=is_consolidated,
                original_start_timestamp=metadata.get('original_start'),
                original_finish_timestamp=metadata.get('original_end'),
                trimmed_start_ns=metadata.get('trimmed_start_ns'),
                trimmed_finish_ns=metadata.get('trimmed_end_ns'),
                boundary_confidence=metadata.get('confidence')
            )

            results.append(result)

        return results