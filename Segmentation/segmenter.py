"""Main task-based segmentation implementation with sliding window approach."""

from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .data_types import TaskSegmentResult, SegmentationConfig
from .signal_processing import (
    ExponentialSmoother, OutlierRejector, InterpolationHelper,
    calculate_gaze_deviation_degrees
)


class TaskSegmenter:
    """Main class for performing task-based session segmentation using sliding window approach."""

    def __init__(self, config: SegmentationConfig):
        """Initialize the task segmenter.

        Args:
            config: Configuration parameters for segmentation
        """
        self.config = config
        self.smoother = ExponentialSmoother(alpha=config.head_filter_alpha)
        self.outlier_rejector = OutlierRejector(
            displacement_threshold_m=config.outlier_displacement_threshold_m,
            time_window_ms=config.outlier_time_window_ms,
            speed_threshold_ms=config.outlier_speed_threshold_ms
        )

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

        # Step 3: Apply signal processing
        positions, timestamps = self._apply_signal_processing(positions, timestamps)

        # Step 4: Extract and process gaze data
        gaze_data = self._process_gaze_data(session_data)

        # Step 5: Extract hand tracking data (if enabled)
        hand_tracking_data = None
        if self.config.require_hand_tracking:
            hand_tracking_data = self._process_hand_tracking_data(session_data)

        # Step 6: Apply sliding window stability detection
        stable_timestamps = self._apply_sliding_window(
            positions, timestamps, gaze_data, hand_tracking_data, session_data
        )

        # Step 7: Create candidate segments from stable periods
        candidate_segments = self._create_candidate_segments(
            stable_timestamps, positions, timestamps
        )

        # Step 8: Apply time buffers
        buffered_segments = self._apply_time_buffers(candidate_segments, session_data)

        # Step 9: Merge consecutive segments
        merged_segments = self._merge_segments(buffered_segments)

        # Step 10: Consolidate locations
        final_segments = self._consolidate_locations(merged_segments)

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

    def _apply_signal_processing(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply smoothing and outlier rejection to position data.

        Returns:
            Tuple of (processed_positions, filtered_timestamps)
        """
        if len(positions) == 0:
            return positions, timestamps

        # Step 1: Remove outliers
        outlier_mask = self.outlier_rejector.reject_outliers(positions, timestamps)
        valid_positions = positions[outlier_mask]
        valid_timestamps = timestamps[outlier_mask]

        if len(valid_positions) == 0:
            return np.array([]), np.array([])

        # Step 2: Apply exponential smoothing
        smoothed_positions = self.smoother.apply_ema(valid_positions)

        return smoothed_positions, valid_timestamps

    def _process_gaze_data(self, session_data) -> Optional[pd.DataFrame]:
        """Process gaze data with filtering and smoothing."""
        if not self.config.require_gaze_stability:
            return None

        if not hasattr(session_data, 'gaze') or session_data.gaze.empty:
            return None

        gaze_df = session_data.gaze.copy()

        # Filter out unwanted gaze states
        if 'gazeState' in gaze_df.columns:
            valid_states = ~gaze_df['gazeState'].isin(self.config.filtered_gaze_states)
            gaze_df = gaze_df[valid_states]

        # Apply exponential moving average to gaze directions
        if all(col in gaze_df.columns for col in ['dirX', 'dirY', 'dirZ']):
            gaze_directions = gaze_df[['dirX', 'dirY', 'dirZ']].values
            if len(gaze_directions) > 0:
                smoothed_directions = self.smoother.apply_ema(gaze_directions)
                gaze_df[['dirX', 'dirY', 'dirZ']] = smoothed_directions

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
        """Apply sliding window stability detection.

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

            # Check all stability criteria within this window
            is_stable = True

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
        """Check if all positions in window are within spatial stability radius."""
        if len(window_positions) < 2:
            return False

        start_position = window_positions[0]

        for position in window_positions[1:]:
            distance = np.linalg.norm(position - start_position)
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
        if all(col in window_gaze.columns for col in ['dirX', 'dirY', 'dirZ']):
            directions = window_gaze[['dirX', 'dirY', 'dirZ']].values

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
                    if all(col in gaze_row for col in ['originX', 'originY', 'originZ']):
                        gaze_origin = np.array([
                            gaze_row['originX'], gaze_row['originY'], gaze_row['originZ']
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
        """Merge segments separated by small gaps."""
        if len(segments) <= 1:
            return segments

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            current_start, current_end, current_start_pos, current_end_pos = current_segment
            next_start, next_end, next_start_pos, next_end_pos = next_segment

            gap_s = (next_start - current_end) / 1e9

            if gap_s <= self.config.merge_consecutive_gap_s:
                # Merge segments
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

    def _generate_results(self,
                        segments: List[Tuple[int, int, np.ndarray, np.ndarray, bool]]) -> List[TaskSegmentResult]:
        """Generate final TaskSegmentResult objects."""
        results = []

        for i, (start_ts, end_ts, start_pos, end_pos, is_consolidated) in enumerate(segments):
            segment_id = f"seg_task_{i+1:03d}"
            duration_ns = end_ts - start_ts

            result = TaskSegmentResult(
                segmentation_id=segment_id,
                start_timestamp=start_ts,
                finish_timestamp=end_ts,
                duration_ns=duration_ns,
                start_location=start_pos,
                finish_location=end_pos,
                is_location_consolidated=is_consolidated
            )

            results.append(result)

        return results