"""Analytics plugins for MLGaze Viewer."""

from src.plugin_sys.base import AnalyticsPlugin
from .aoi_analyzer import AOIAnalyzer
from .object_detector import ObjectDetector
from .gaze_object_interaction import GazeObjectInteraction
from .gaze_3d_heatmap import Gaze3DHeatmap
from .gaze_3d_clustering import Gaze3DClustering
from .object_instance_tracker import ObjectInstanceTracker
from .task_segmentation import TaskSegmentationPlugin

__all__ = [
    "AnalyticsPlugin",
    "AOIAnalyzer",
    "ObjectDetector",
    "GazeObjectInteraction",
    "Gaze3DHeatmap",
    "Gaze3DClustering",
    "ObjectInstanceTracker",
    "TaskSegmentationPlugin"
]


def load_plugins(plugin_names: list) -> list:
    """Load analytics plugins by name.

    Args:
        plugin_names: List of plugin class names to load

    Returns:
        List of instantiated plugin objects
    """
    plugins = []

    for name in plugin_names:
        # Updated to use class names consistently
        if name == "AOIAnalyzer":
            plugins.append(AOIAnalyzer())
        elif name == "ObjectDetector":
            plugins.append(ObjectDetector())
        elif name == "GazeObjectInteraction":
            plugins.append(GazeObjectInteraction())
        elif name == "Gaze3DHeatmap":
            plugins.append(Gaze3DHeatmap())
        elif name == "Gaze3DClustering":
            plugins.append(Gaze3DClustering())
        elif name == "ObjectInstanceTracker":
            plugins.append(ObjectInstanceTracker())
        elif name == "TaskSegmentationPlugin":
            plugins.append(TaskSegmentationPlugin())
        # Add more plugins here as they are implemented

    return plugins