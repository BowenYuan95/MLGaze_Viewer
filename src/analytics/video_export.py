"""Video Export plugin for generating MP4 videos from session frames."""

from typing import Dict, List, Any, Optional
from pathlib import Path

from src.plugin_sys.base import AnalyticsPlugin
from src.core import SessionData
from src.utils.video_generator import VideoGenerator


class VideoExportPlugin(AnalyticsPlugin):
    """Plugin for exporting session camera frames as MP4 videos.

    This plugin automatically generates MP4 videos for each camera in the session
    during visualization. Videos are saved to session_dir/videos/ directory.

    Configuration options:
    - enabled: Enable/disable video export (default: True)
    - overlay_gaze: Include gaze overlay in videos (default: False)
    - fps_override: Override auto-detected FPS (default: None)
    - codec: Video codec to use (default: 'mp4v')
    - cameras: List of specific cameras to export (default: all)
    """

    def __init__(self):
        """Initialize video export plugin."""
        super().__init__("VideoExport")
        self.generator: Optional[VideoGenerator] = None
        self.generated_videos: Dict[str, Path] = {}

    def get_dependencies(self) -> List[str]:
        """No dependencies - works directly with session data."""
        return []

    def get_optional_dependencies(self) -> List[str]:
        """No optional dependencies."""
        return []

    def validate_data(self, session: SessionData) -> bool:
        """Check if the session data is valid for video export.

        Args:
            session: SessionData to validate

        Returns:
            True if data is valid, False otherwise
        """
        validation_issues = []

        # Check for camera data
        if not session.camera_metadata:
            validation_issues.append("No camera metadata available")

        # Check for frames (at least one camera should have frames)
        if not session.frames:
            validation_issues.append("No camera frames available")

        # Check input directory exists
        if not session.input_directory:
            validation_issues.append("No input directory specified")
        else:
            cameras_dir = Path(session.input_directory) / "cameras"
            if not cameras_dir.exists():
                validation_issues.append(f"Cameras directory not found: {cameras_dir}")

        if validation_issues:
            self.logger.error(f"Data validation failed: {validation_issues}")
            return False

        return True

    def process(self, session: SessionData, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process session data to generate MP4 videos.

        Args:
            session: SessionData containing camera frames
            config: Optional configuration dictionary with plugin settings

        Returns:
            Dictionary containing:
                - generated_videos: Dict mapping camera names to video paths
                - total_videos: Number of videos generated
                - output_directory: Path to videos directory
        """
        # Extract video export configuration
        if config and 'video_export' in config:
            export_config = config['video_export']
        else:
            # Use defaults
            self.logger.warning("No video_export config found, using defaults")
            export_config = {}

        # Get configuration parameters
        overlay_gaze = export_config.get('overlay_gaze', False)
        fps_override = export_config.get('fps_override', None)
        codec = export_config.get('codec', 'mp4v')
        camera_filter = export_config.get('cameras', None)

        self.logger.info("Starting video export")
        self.logger.info(f"  Overlay gaze: {overlay_gaze}")
        self.logger.info(f"  FPS override: {fps_override if fps_override else 'auto-detect'}")
        self.logger.info(f"  Codec: {codec}")

        # Validate session data
        if not self.validate_data(session):
            error_result = {
                'error': 'Session data validation failed',
                'generated_videos': {},
                'total_videos': 0,
                'output_directory': None
            }
            session.set_plugin_result(self.__class__.__name__, error_result)
            return error_result

        # Create video generator
        self.generator = VideoGenerator(codec=codec)

        # Determine output directory
        output_dir = Path(session.input_directory) / "videos"

        # Generate videos
        self.generated_videos = self.generator.generate_all_videos(
            session_dir=Path(session.input_directory),
            output_dir=output_dir,
            fps_override=fps_override,
            overlay_gaze=overlay_gaze,
            camera_filter=camera_filter
        )

        # Calculate metrics
        total_videos = len(self.generated_videos)

        self.logger.info(f"Video export complete: {total_videos} videos generated")

        # Prepare results
        results = {
            'generated_videos': {name: str(path) for name, path in self.generated_videos.items()},
            'total_videos': total_videos,
            'output_directory': str(output_dir)
        }

        # Store results in session
        session.set_plugin_result(self.__class__.__name__, results)

        return results

    def get_summary(self, results: Dict) -> str:
        """Generate a text summary of the video export results.

        Args:
            results: Video export results from process method

        Returns:
            Human-readable summary string
        """
        if 'error' in results:
            return f"Video Export failed: {results['error']}"

        total_videos = results.get('total_videos', 0)
        output_dir = results.get('output_directory', 'unknown')

        summary = f"Video Export: {total_videos} video(s) generated\n"
        summary += f"  Output directory: {output_dir}\n"

        # Add video details if available
        videos = results.get('generated_videos', {})
        if videos:
            summary += "\n  Generated videos:\n"
            for camera_name, video_path in videos.items():
                video_file = Path(video_path)
                if video_file.exists():
                    size_mb = video_file.stat().st_size / (1024 * 1024)
                    summary += f"    {camera_name}: {video_file.name} ({size_mb:.2f} MB)\n"
                else:
                    summary += f"    {camera_name}: {video_file.name}\n"

        return summary
