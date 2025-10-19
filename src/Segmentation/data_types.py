"""Data type definitions for task-based session segmentation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TaskSegmentResult:
    """Result of task-based session segmentation.

    Contains the exact 6 fields specified for segmentation output:
    - segmentation_id: Unique identifier (format: "seg_task_001")
    - start_timestamp: Segment start time (nanoseconds)
    - finish_timestamp: Segment end time (nanoseconds)
    - duration_ns: Time period (finish - start)
    - start_location: 3D position at start [x, y, z]
    - finish_location: 3D position at end [x, y, z]
    - is_location_consolidated: True if locations were consolidated (< 0.5m apart)
    """
    segmentation_id: str              # Format: "seg_task_001"
    start_timestamp: int              # Nanoseconds
    finish_timestamp: int             # Nanoseconds
    duration_ns: int                  # Time period (finish - start)
    start_location: np.ndarray        # [x, y, z] in Unity coordinates
    finish_location: np.ndarray       # [x, y, z] in Unity coordinates
    is_location_consolidated: bool    # True if distance < consolidation threshold

    # DAG support for tracking dependencies between segments
    dependency_segment_ids: Optional[List[str]] = None  # Parent segment IDs this depends on
    edge_costs: Optional[Dict[str, Dict[str, float]]] = None  # Cost metrics per dependency edge

    def __post_init__(self):
        """Validate segment properties after creation."""
        # Initialize DAG fields if None
        if self.dependency_segment_ids is None:
            self.dependency_segment_ids = []
        if self.edge_costs is None:
            self.edge_costs = {}

        # Validate edge_costs keys match dependency_segment_ids
        if self.edge_costs:
            for parent_id in self.edge_costs.keys():
                if parent_id not in self.dependency_segment_ids:
                    raise ValueError(
                        f"edge_costs key '{parent_id}' not found in dependency_segment_ids. "
                        f"All edge_costs keys must correspond to entries in dependency_segment_ids."
                    )

        # Validate timestamp range
        if self.finish_timestamp <= self.start_timestamp:
            raise ValueError(f"Invalid timestamp range: {self.finish_timestamp} <= {self.start_timestamp}")

        # Ensure duration is correctly calculated
        calculated_duration = self.finish_timestamp - self.start_timestamp
        if self.duration_ns != calculated_duration:
            self.duration_ns = calculated_duration

        # Validate location arrays
        if len(self.start_location) != 3 or len(self.finish_location) != 3:
            raise ValueError("Locations must be 3D coordinates [x, y, z]")
    
    @property
    def duration_s(self) -> float:
        """Get duration in seconds for convenience."""
        return self.duration_ns / 1e9
    
    @property
    def location_distance(self) -> float:
        """Calculate distance between start and finish locations in meters."""
        return float(np.linalg.norm(self.finish_location - self.start_location))


@dataclass
class SegmentationConfig:
    """Configuration parameters for task-based segmentation.

    Contains all adjustable parameters for the segmentation algorithm.
    Values can be overridden via the global configuration system.
    """
    # Optional criteria (all can be enabled/disabled)
    require_spatial_stability: bool = True
    require_gaze_stability: bool = True
    require_hand_tracking: bool = False

    # Spatial stability criteria
    spatial_stability_radius_m: float = 0.3
    spatial_stability_min_duration_s: float = 1.0
    stability_time_window_s: float = 1.5
    stability_buffer_start_s: float = 0.5
    stability_buffer_end_s: float = 0.5

    # Gaze analysis parameters
    gaze_deviation_threshold_deg: float = 20.0
    focus_distance_threshold_m: float = 1.0
    gaze_smoothing_window_s: float = 1.5
    gaze_sampling_interval_s: float = 0.5
    filtered_gaze_states: List[str] = None

    # Signal processing parameters
    head_filter_alpha: float = 0.48
    outlier_displacement_threshold_m: float = 1.5
    outlier_time_window_ms: float = 100.0
    outlier_speed_threshold_ms: float = 3.0

    # Segment processing
    merge_consecutive_gap_s: float = 1.5
    location_consolidation_threshold_m: float = 0.5
    min_segment_duration_s: float = 1.0

    # Stability detection mode
    stability_mode: str = "percentage"  # "strict" or "percentage"

    # Percentage-based stability thresholds (only used when mode="percentage")
    head_instability_threshold_percent: float = 20.0
    gaze_instability_threshold_percent: float = 20.0
    hand_untracked_threshold_percent: float = 20.0
    min_samples_for_percentage_mode: int = 5
    
    def __post_init__(self):
        """Initialize default values and validate parameters."""
        if self.filtered_gaze_states is None:
            self.filtered_gaze_states = ["Blink", "Unknown"]
        
        # Validate critical parameters
        if self.spatial_stability_radius_m <= 0:
            raise ValueError("Spatial stability radius must be positive")
        
        if self.spatial_stability_min_duration_s <= 0:
            raise ValueError("Minimum duration must be positive")
        
        if not (0 <= self.gaze_deviation_threshold_deg <= 180):
            raise ValueError("Gaze deviation threshold must be between 0 and 180 degrees")
        
        if not (0 < self.head_filter_alpha <= 1):
            raise ValueError("Head filter alpha must be between 0 and 1")
    
    @classmethod
    def from_plugin_config(cls, plugin_config: Dict[str, Any]) -> 'SegmentationConfig':
        """Create configuration from plugin config dictionary."""
        return cls(
            require_spatial_stability=plugin_config.get('require_spatial_stability', True),
            require_gaze_stability=plugin_config.get('require_gaze_stability', True),
            require_hand_tracking=plugin_config.get('require_hand_tracking', False),
            spatial_stability_radius_m=plugin_config.get('spatial_stability_radius_m', 0.3),
            spatial_stability_min_duration_s=plugin_config.get('spatial_stability_min_duration_s', 1.0),
            stability_time_window_s=plugin_config.get('stability_time_window_s', 1.5),
            stability_buffer_start_s=plugin_config.get('stability_buffer_start_s', 0.5),
            stability_buffer_end_s=plugin_config.get('stability_buffer_end_s', 0.5),
            gaze_deviation_threshold_deg=plugin_config.get('gaze_deviation_threshold_deg', 20.0),
            focus_distance_threshold_m=plugin_config.get('focus_distance_threshold_m', 1.0),
            gaze_smoothing_window_s=plugin_config.get('gaze_smoothing_window_s', 1.5),
            gaze_sampling_interval_s=plugin_config.get('gaze_sampling_interval_s', 0.5),
            filtered_gaze_states=plugin_config.get('filtered_gaze_states', ["Blink", "Unknown"]),
            head_filter_alpha=plugin_config.get('head_filter_alpha', 0.48),
            outlier_displacement_threshold_m=plugin_config.get('outlier_displacement_threshold_m', 1.5),
            outlier_time_window_ms=plugin_config.get('outlier_time_window_ms', 100.0),
            outlier_speed_threshold_ms=plugin_config.get('outlier_speed_threshold_ms', 3.0),
            merge_consecutive_gap_s=plugin_config.get('merge_consecutive_gap_s', 1.5),
            location_consolidation_threshold_m=plugin_config.get('location_consolidation_threshold_m', 0.5),
            min_segment_duration_s=plugin_config.get('min_segment_duration_s', 1.0),
            stability_mode=plugin_config.get('stability_mode', 'percentage'),
            head_instability_threshold_percent=plugin_config.get('head_instability_threshold_percent', 20.0),
            gaze_instability_threshold_percent=plugin_config.get('gaze_instability_threshold_percent', 20.0),
            hand_untracked_threshold_percent=plugin_config.get('hand_untracked_threshold_percent', 20.0),
            min_samples_for_percentage_mode=plugin_config.get('min_samples_for_percentage_mode', 5)
        )