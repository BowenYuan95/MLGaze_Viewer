"""Signal processing utilities for task-based session segmentation."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


class ExponentialSmoother:
    """Exponential moving average smoother for filtering noisy sensor data."""

    def __init__(self, alpha: float = 0.48):
        """Initialize the exponential smoother.

        Args:
            alpha: Smoothing factor (0 < α ≤ 1). Higher values = less smoothing.
                  Default 0.48 matches Unity implementation for head filtering.
        """
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")

        self.alpha = alpha
        self._previous_value: Optional[np.ndarray] = None

    def apply_ema(self, values: np.ndarray) -> np.ndarray:
        """Apply exponential moving average to a sequence of values.

        Args:
            values: Array of shape (N, D) where N is number of samples and D is dimensions

        Returns:
            Smoothed array of same shape
        """
        if len(values) == 0:
            return values

        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]  # First value unchanged

        for i in range(1, len(values)):
            smoothed[i] = self.alpha * values[i] + (1 - self.alpha) * smoothed[i-1]

        return smoothed

    def reset(self):
        """Reset the smoother state."""
        self._previous_value = None


class OutlierRejector:
    """Detects and removes outliers from position and movement data."""

    def __init__(self,
                 displacement_threshold_m: float = 1.5,
                 time_window_ms: float = 100.0,
                 speed_threshold_ms: float = 3.0):
        """Initialize the outlier rejector.

        Args:
            displacement_threshold_m: Maximum allowed displacement per time window (meters)
            time_window_ms: Time window for displacement check (milliseconds)
            speed_threshold_ms: Maximum allowed speed (m/s)
        """
        self.displacement_threshold_m = displacement_threshold_m
        self.time_window_ms = time_window_ms
        self.speed_threshold_ms = speed_threshold_ms

    def reject_by_displacement(self,
                              positions: np.ndarray,
                              timestamps: np.ndarray) -> np.ndarray:
        """Remove samples with excessive displacement within time windows.

        Args:
            positions: Array of 3D positions, shape (N, 3)
            timestamps: Array of timestamps in nanoseconds, shape (N,)

        Returns:
            Boolean mask where True = keep sample, False = reject
        """
        if len(positions) <= 1:
            return np.ones(len(positions), dtype=bool)

        keep_mask = np.ones(len(positions), dtype=bool)

        for i in range(1, len(positions)):
            # Calculate displacement from previous sample
            displacement = np.linalg.norm(positions[i] - positions[i-1])

            # Calculate time difference in milliseconds
            time_diff_ms = (timestamps[i] - timestamps[i-1]) / 1e6

            # Check if displacement exceeds threshold for this time window
            if time_diff_ms > 0 and time_diff_ms <= self.time_window_ms:
                if displacement > self.displacement_threshold_m:
                    keep_mask[i] = False

        return keep_mask

    def reject_by_speed(self,
                       positions: np.ndarray,
                       timestamps: np.ndarray) -> np.ndarray:
        """Remove samples with excessive movement speed.

        Args:
            positions: Array of 3D positions, shape (N, 3)
            timestamps: Array of timestamps in nanoseconds, shape (N,)

        Returns:
            Boolean mask where True = keep sample, False = reject
        """
        if len(positions) <= 1:
            return np.ones(len(positions), dtype=bool)

        keep_mask = np.ones(len(positions), dtype=bool)

        for i in range(1, len(positions)):
            # Calculate displacement and time difference
            displacement = np.linalg.norm(positions[i] - positions[i-1])
            time_diff_s = (timestamps[i] - timestamps[i-1]) / 1e9

            # Calculate speed (m/s)
            if time_diff_s > 0:
                speed = displacement / time_diff_s

                if speed > self.speed_threshold_ms:
                    keep_mask[i] = False

        return keep_mask

    def reject_outliers(self,
                       positions: np.ndarray,
                       timestamps: np.ndarray) -> np.ndarray:
        """Apply both displacement and speed outlier rejection.

        Args:
            positions: Array of 3D positions, shape (N, 3)
            timestamps: Array of timestamps in nanoseconds, shape (N,)

        Returns:
            Boolean mask where True = keep sample, False = reject
        """
        displacement_mask = self.reject_by_displacement(positions, timestamps)
        speed_mask = self.reject_by_speed(positions, timestamps)

        # Keep samples that pass both tests
        return displacement_mask & speed_mask


class InterpolationHelper:
    """Handles missing data interpolation and gap management."""

    @staticmethod
    def zero_order_hold(data: pd.DataFrame,
                       timestamp_col: str = 'timestamp',
                       max_gap_ms: float = 100.0) -> pd.DataFrame:
        """Apply zero-order hold interpolation for small gaps.

        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column (nanoseconds)
            max_gap_ms: Maximum gap size to interpolate (milliseconds)

        Returns:
            DataFrame with interpolated values
        """
        if data.empty:
            return data

        # Sort by timestamp
        data_sorted = data.sort_values(timestamp_col).copy()

        # Calculate gaps between consecutive samples
        time_diffs_ms = data_sorted[timestamp_col].diff() / 1e6

        # Identify gaps that need interpolation
        small_gaps = (time_diffs_ms > 0) & (time_diffs_ms <= max_gap_ms)

        if not small_gaps.any():
            return data_sorted

        # Apply forward fill for small gaps
        for col in data_sorted.columns:
            if col != timestamp_col:
                data_sorted[col] = data_sorted[col].ffill()

        return data_sorted

    @staticmethod
    def identify_large_gaps(timestamps: np.ndarray,
                           gap_threshold_ms: float = 300.0) -> np.ndarray:
        """Identify large gaps that should be excluded from analysis.

        Args:
            timestamps: Array of timestamps in nanoseconds
            gap_threshold_ms: Threshold for large gaps (milliseconds)

        Returns:
            Boolean mask where True = valid data, False = in large gap
        """
        if len(timestamps) <= 1:
            return np.ones(len(timestamps), dtype=bool)

        # Calculate time differences in milliseconds
        time_diffs_ms = np.diff(timestamps) / 1e6

        # Find large gaps
        large_gaps = time_diffs_ms > gap_threshold_ms

        # Create mask - mark samples after large gaps as invalid
        valid_mask = np.ones(len(timestamps), dtype=bool)

        gap_indices = np.where(large_gaps)[0]
        for gap_idx in gap_indices:
            # Mark the sample after the gap as invalid
            # (this creates exclusion zones for large gaps)
            if gap_idx + 1 < len(valid_mask):
                valid_mask[gap_idx + 1] = False

        return valid_mask


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