"""Task Segmentation plugin for identifying task-based segments in session data."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from src.plugin_sys.base import AnalyticsPlugin
from src.core import SessionData
from src.Segmentation.data_types import SegmentationConfig, TaskSegmentResult
from src.Segmentation.segmenter import TaskSegmenter
from src.Segmentation.export import export_task_segments


class TaskSegmentationPlugin(AnalyticsPlugin):
    """Plugin for task-based segmentation of session data.

    This plugin identifies stable task periods in the session data based on:
    - Spatial stability (head position)
    - Gaze stability (focus direction)
    - Hand tracking (optional)

    The plugin uses a sliding window approach to detect periods where the user
    is performing focused tasks at specific locations.
    """

    def __init__(self):
        """Initialize task segmentation plugin."""
        super().__init__("TaskSegmentation")
        self.segmenter: Optional[TaskSegmenter] = None
        self.segments: List[TaskSegmentResult] = []

    def get_dependencies(self) -> List[str]:
        """No dependencies - works directly with session data."""
        return []

    def get_optional_dependencies(self) -> List[str]:
        """No optional dependencies."""
        return []

    def validate_data(self, session: SessionData) -> bool:
        """Check if the session data is valid for segmentation.

        Args:
            session: SessionData to validate

        Returns:
            True if data is valid, False otherwise
        """
        validation_issues = []

        # Check for camera data
        if not session.frames:
            validation_issues.append("No camera frames available")

        # Check for primary camera
        if not hasattr(session, 'primary_camera') or session.primary_camera is None:
            validation_issues.append("No primary camera specified")

        # Check for camera metadata
        primary_cam = session.primary_camera if hasattr(session, 'primary_camera') else None
        if primary_cam and primary_cam not in session.frames:
            validation_issues.append(f"Primary camera '{primary_cam}' not found in frames")

        if validation_issues:
            self.logger.error(f"Data validation failed: {validation_issues}")
            return False

        return True

    def get_required_columns(self) -> Dict[str, list]:
        """Get required columns for segmentation.

        Returns:
            Dictionary mapping dataframe names to required column lists
        """
        return {
            'camera_metadata': [
                'frameId', 'deviceTimestamp_ns',
                'position_x', 'position_y', 'position_z'
            ],
            'gaze_data': [
                'timestamp_ns', 'gazeState',
                'gazeOriginWorld_x', 'gazeOriginWorld_y', 'gazeOriginWorld_z',
                'gazeDirectionWorld_x', 'gazeDirectionWorld_y', 'gazeDirectionWorld_z'
            ]
        }

    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process session data to identify task segments.

        Args:
            session: SessionData containing camera, gaze, and optional hand tracking
            config: Optional configuration dictionary with plugin settings

        Returns:
            Dictionary containing:
                - segments: List of TaskSegmentResult objects
                - total_segments: Number of detected segments
                - total_duration_s: Total duration of all segments
                - config: Configuration used for segmentation
        """
        # Extract task_segmenter configuration
        if config and 'task_segmenter' in config:
            task_config_dict = config['task_segmenter']
        else:
            # Use defaults from SegmentationConfig
            self.logger.warning("No task_segmenter config found, using defaults")
            task_config_dict = {}

        # Create SegmentationConfig from dictionary
        seg_config = SegmentationConfig(**task_config_dict)

        self.logger.info(f"Starting task segmentation with mode={seg_config.stability_mode}")
        self.logger.info(f"  Spatial stability: {seg_config.require_spatial_stability}")
        self.logger.info(f"  Gaze stability: {seg_config.require_gaze_stability}")
        self.logger.info(f"  Hand tracking: {seg_config.require_hand_tracking}")

        # Validate session data
        if not self.validate_data(session):
            error_result = {
                'error': 'Session data validation failed',
                'segments': [],
                'total_segments': 0,
                'total_duration_s': 0.0,
                'config': seg_config.__dict__
            }
            session.set_plugin_result(self.__class__.__name__, error_result)
            return error_result

        # Create segmenter and process
        self.segmenter = TaskSegmenter(seg_config)
        self.segments = self.segmenter.segment(session)

        # Calculate metrics
        total_segments = len(self.segments)
        total_duration_s = sum(seg.duration_ns / 1e9 for seg in self.segments)

        self.logger.info(f"Segmentation complete: {total_segments} segments found")
        self.logger.info(f"  Total task duration: {total_duration_s:.2f}s")

        # Prepare results
        results = {
            'segments': self.segments,
            'total_segments': total_segments,
            'total_duration_s': total_duration_s,
            'config': seg_config.__dict__
        }

        # Store results in session
        session.set_plugin_result(self.__class__.__name__, results)

        # Export segments to session directory
        if self.segments:
            self._export_segments(session)
        else:
            self.logger.info("No segments to export")

        return results

    def _export_segments(self, session: SessionData) -> None:
        """Export segmentation results to files in the session directory.

        Args:
            session: SessionData containing input directory path
        """
        # Export directly to the session's input directory
        if not session.input_directory:
            self.logger.warning("No input directory in session, skipping export")
            return

        export_path = Path(session.input_directory) / "Segmented_Tasks"
        export_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting {len(self.segments)} segments to {export_path}")

        # Export using the export module
        export_task_segments(
            segments=self.segments,
            output_dir=str(export_path),
            formats=['csv', 'json', 'summary']
        )

        self.logger.info(f"Export complete: {export_path}")

    def get_summary(self, results: Dict) -> str:
        """Generate a text summary of the segmentation results.

        Args:
            results: Segmentation results from process method

        Returns:
            Human-readable summary string
        """
        if 'error' in results:
            return f"Task Segmentation failed: {results['error']}"

        total_segments = results.get('total_segments', 0)
        total_duration = results.get('total_duration_s', 0.0)

        summary = f"Task Segmentation: {total_segments} segments detected\n"
        summary += f"  Total task duration: {total_duration:.2f}s\n"

        # Add segment details if available
        segments = results.get('segments', [])
        if segments:
            avg_duration = total_duration / total_segments if total_segments > 0 else 0
            summary += f"  Average segment duration: {avg_duration:.2f}s\n"

            # Show first few segments
            summary += "\n  First segments:\n"
            for i, seg in enumerate(segments[:3]):
                duration = seg.duration_ns / 1e9
                summary += f"    {seg.segmentation_id}: {duration:.2f}s at "
                summary += f"({seg.start_location[0]:.2f}, {seg.start_location[1]:.2f}, {seg.start_location[2]:.2f})\n"

        return summary
