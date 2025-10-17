"""Configuration management for MLGaze Viewer."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import json


@dataclass
class VisualizationConfig:
    """Configuration settings for gaze visualization."""
    
    input_directory: str = "input"
    output_directory: str = "output"
    
    # Multi-camera settings
    primary_camera: str = ""  # Selected primary camera for 3D visualization
    
    # Timestamp synchronization settings
    timestamp_sync_mode: str = "union"  # "union" (first-to-last) or "intersection" (all-to-any)
    
    enable_fade_trail: bool = True
    fade_duration: float = 5.0
    
    # Sliding window settings
    enable_sliding_window: bool = False
    sliding_window_duration: float = 10.0  # seconds
    sliding_window_update_rate: float = 0.5  # seconds between updates
    sliding_window_3d_gaze: bool = True  # Apply to 3D gaze points
    sliding_window_3d_trajectory: bool = True  # Apply to 3D trajectories
    sliding_window_camera: bool = True  # Apply to camera positions
    
    # Visualization toggles
    show_point_cloud: bool = True
    show_gaze_trajectory: bool = True
    show_camera_trajectory: bool = True
    color_by_gaze_state: bool = True
    test_y_flip: bool = False
    show_coordinate_indicators: bool = True
    
    show_imu_data: bool = True
    
    # Object detection settings
    enable_object_detection: bool = False
    object_detection_model: str = "base"  # nano, small, medium, base, custom
    object_detection_confidence: float = 0.5  # 0.0-1.0
    object_detection_device: str = "auto"  # auto, cpu, cuda, mps
    
    # Custom model settings
    object_detection_custom_model_path: str = ""  # Path to custom fine-tuned model
    object_detection_custom_classes: str = ""  # Comma-separated custom class names
    
    # Detection quality settings
    object_detection_nms_threshold: float = 0.5  # Non-Maximum Suppression threshold
    object_detection_target_classes: str = ""  # Comma-separated target classes (empty = all)
    
    # Image preprocessing settings (for accuracy improvement)
    object_detection_preprocessing_mode: str = "center_crop"  # "none", "center_crop", "padding"
    object_detection_preserve_aspect_ratio: bool = True  # For future use/reference
    
    # Object detection model management
    prefer_local_models: bool = True  # Use local models first before downloading
    auto_download_models: bool = True  # Allow automatic model downloading
    
    # Entity paths for Rerun
    entity_paths: Dict[str, str] = field(default_factory=lambda: {
        "world": "/world",
        "gaze": "/world/gaze",
        "camera": "/world/camera",
        "sensors": "/sensors",
        "analytics": "/analytics"
    })
    
    # Analytics settings
    enabled_plugins: List[str] = field(default_factory=lambda: [
        "fixation_detector",
        "Gaze3DHeatmap",
        "Gaze3DClustering",
        "ObjectInstanceTracker",
        "TaskSegmentationPlugin"
    ])
    
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "fixation_detector": {
            "velocity_threshold": 30,  # degrees/sec
            "min_duration": 100,  # milliseconds
            "max_dispersion": 1.0  # degrees
        },
        "aoi_analyzer": {
            "regions": []  # Will be populated at runtime
        },
        "object_detector": {
            "model_size": "base",
            "confidence_threshold": 0.5,
            "device": "auto",
            "cache_detections": True
        },
        "dwell_analyzer": {
            "min_dwell_time": 500,  # milliseconds
            "merge_threshold": 50  # milliseconds
        },
        "Gaze3DHeatmap": {
            "enabled": True,
            "voxel_size": 0.1,  # 10cm voxels
            "min_density": 5,  # Min points per voxel to display
            "show_heatmap": True,  # Toggle visualization
            "use_boxes": True,  # Use Boxes3D instead of Points3D
            "opacity_min": 0.3,  # Minimum opacity
            "opacity_max": 1.0,  # Maximum opacity
            "fill_mode": "solid",  # solid or wireframe
            "filter_enabled": False,  # Enable gaze state filtering
            "gaze_states_filter": []  # List of gaze states to include (empty = all)
        },
        "Gaze3DClustering": {
            "enabled": True,
            "min_cluster_size": 30,  # Min points to form cluster
            "min_samples": 5,  # Core point density requirement
            "epsilon_m": 0.15,  # Max distance in meters
            "show_bounds": True,  # Show bounding boxes
            "bound_opacity": 0.3,  # Bounding box opacity
            "point_opacity": 0.8,  # Cluster points opacity
            "filter_enabled": False,  # Enable gaze state filtering
            "gaze_states_filter": []  # List of gaze states to include (empty = all)
        },
        "ObjectInstanceTracker": {
            "enabled": True,
            "iou_threshold": 0.3,  # IoU threshold for matching objects across frames
            "max_frame_gap": 10,  # Max frames between detections before considering instance lost
            "min_confidence": 0.5,  # Minimum detection confidence to track
            "min_gaze_points": 50,  # Min gaze points for 3D bbox calculation
            "gaze_time_window_ms": 100,  # Time window for gaze aggregation (ms)
            "generate_reports": True,  # Generate CSV reports
            "show_3d_boxes": True,  # Show 3D bounding boxes in Rerun
            "box_opacity": 0.4,  # 3D bounding box opacity
            "instance_colors": True,  # Use per-instance colors
            "cluster_distance_m": 0.5  # Max distance to associate with gaze cluster
        }
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, file_path: Path) -> None:
        """Save configuration to YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, file_path: Path) -> 'VisualizationConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, file_path: Path) -> 'VisualizationConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizationConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        return self.plugin_configs.get(plugin_name, {})
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific plugin."""
        if plugin_name not in self.plugin_configs:
            self.plugin_configs[plugin_name] = {}
        self.plugin_configs[plugin_name].update(config)
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        return plugin_name in self.enabled_plugins
    
    def sync_object_detection_config(self) -> None:
        """Sync object detection UI settings with plugin config."""
        if "object_detector" not in self.plugin_configs:
            self.plugin_configs["object_detector"] = {}
        
        # Parse custom classes and target classes
        custom_classes = [cls.strip() for cls in self.object_detection_custom_classes.split(',') if cls.strip()] if self.object_detection_custom_classes else None
        target_classes = [cls.strip() for cls in self.object_detection_target_classes.split(',') if cls.strip()] if self.object_detection_target_classes else None
        
        # Update plugin config from UI settings
        self.plugin_configs["object_detector"].update({
            "model_size": self.object_detection_model,
            "confidence_threshold": self.object_detection_confidence,
            "device": self.object_detection_device,
            "custom_model_path": self.object_detection_custom_model_path if self.object_detection_model == "custom" else None,
            "custom_class_names": custom_classes,
            "nms_threshold": self.object_detection_nms_threshold,
            "target_classes": target_classes,
            "prefer_local_models": self.prefer_local_models,
            "auto_download_models": self.auto_download_models,
            "preprocessing_mode": self.object_detection_preprocessing_mode,
            "preserve_aspect_ratio": self.object_detection_preserve_aspect_ratio
        })
        
        # Update enabled plugins list
        if self.enable_object_detection:
            if "object_detector" not in self.enabled_plugins:
                self.enabled_plugins.append("object_detector")
        else:
            if "object_detector" in self.enabled_plugins:
                self.enabled_plugins.remove("object_detector")