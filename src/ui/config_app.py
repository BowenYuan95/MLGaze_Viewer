"""
MLGaze Viewer TUI Configuration App

Textual-based terminal UI for configuring visualization parameters.
"""

from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Input, Checkbox, Select, DirectoryTree, RadioSet, RadioButton,
    SelectionList
)
from textual.widgets.selection_list import Selection
from textual.screen import ModalScreen

# Import the centralized config from core module
from src.core.config import VisualizationConfig
from src.core import SessionMetadata

# Import for object detection model availability
try:
    from src.analytics.object_detector import ObjectDetector
    OBJECT_DETECTION_AVAILABLE = True
except ImportError:
    OBJECT_DETECTION_AVAILABLE = False
    ObjectDetector = None


class DirectoryBrowserScreen(ModalScreen[str]):
    """Modal screen for directory selection."""
    
    CSS = """
    DirectoryBrowserScreen {
        background: $surface;
    }
    
    .directory-container {
        width: 80%;
        height: 80%;
        margin: 2;
        padding: 1;
        border: solid $primary;
    }
    
    .browser-buttons {
        margin-top: 1;
        text-align: center;
    }
    
    .current-path {
        margin-bottom: 1;
        text-style: bold;
        color: $primary;
    }
    """
    
    def __init__(self, initial_path: str = ".") -> None:
        super().__init__()
        self.initial_path = Path(initial_path).resolve()
        self.selected_path = self.initial_path
    
    def compose(self) -> ComposeResult:
        with Container(classes="directory-container"):
            yield Static("Select Directory", classes="section-title")
            
            # Navigation controls
            with Horizontal(classes="nav-controls"):
                yield Button("Parent Folder", id="parent_dir")
                yield Button("Home", id="home_dir")
                yield Static(f"Current: {self.initial_path}", id="current_path_display")
            
            yield Static(f"Selected: {self.selected_path}", classes="current-path", id="current_path")
            yield DirectoryTree(str(self.initial_path), id="directory_tree")
            
            with Container(classes="browser-buttons"):
                yield Button("Select", variant="primary", id="select_dir")
                yield Button("Cancel", variant="default", id="cancel_dir")
    
    @on(DirectoryTree.NodeHighlighted)
    def update_selection(self, event: DirectoryTree.NodeHighlighted) -> None:
        """Update selected path when user navigates."""
        try:
            self.app.log(f"Node highlighted: {event.node}")
            if hasattr(event.node, 'data') and event.node.data and hasattr(event.node.data, 'path'):
                self.selected_path = Path(event.node.data.path)
                self.app.log(f"Selected path updated: {self.selected_path}")
                # Update the display
                self.query_one("#current_path", Static).update(f"Selected: {self.selected_path}")
            else:
                # Fallback: try to get path from tree's current location
                tree = self.query_one(DirectoryTree)
                if hasattr(tree, 'path'):
                    self.selected_path = Path(tree.path)
                    self.query_one("#current_path", Static).update(f"Selected: {self.selected_path}")
                    self.app.log(f"Fallback path: {self.selected_path}")
        except Exception as e:
            self.app.log(f"Selection update error: {e}")

    
    @on(Button.Pressed, "#select_dir")
    def select_directory(self) -> None:
        """Handle Select button press."""
        try:
            path_to_select = str(self.selected_path)
            self.app.log(f"Selecting directory via button: {path_to_select}")
            self.dismiss(path_to_select)
        except Exception as e:
            self.app.log(f"Button selection error: {e}")
            # Fallback: dismiss with None to avoid crash
            self.dismiss(None)
    
    @on(Button.Pressed, "#cancel_dir")
    def cancel_selection(self) -> None:
        """Handle Cancel button press."""
        self.dismiss(None)
    
    @on(Button.Pressed, "#parent_dir")
    def go_to_parent(self) -> None:
        """Navigate to parent directory."""
        try:
            tree = self.query_one(DirectoryTree)
            current_path = Path(tree.path)
            parent_path = current_path.parent
            
            # Check if we can actually go up (not at filesystem root)
            if parent_path != current_path:
                self.refresh_tree(str(parent_path))
            else:
                self.app.log("Already at filesystem root")
        except Exception as e:
            self.app.log(f"Parent navigation error: {e}")
    
    @on(Button.Pressed, "#home_dir")
    def go_to_home(self) -> None:
        """Navigate to home directory."""
        try:
            home_path = Path.home()
            self.refresh_tree(str(home_path))
        except Exception as e:
            self.app.log(f"Home navigation error: {e}")
    
    def refresh_tree(self, new_path: str) -> None:
        """Refresh directory tree with new path."""
        try:
            new_path_obj = Path(new_path).resolve()
            
            # Update the tree's path
            tree = self.query_one(DirectoryTree)
            tree.path = str(new_path_obj)
            tree.reload()
            
            # Update selected path
            self.selected_path = new_path_obj
            
            # Update displays
            self.query_one("#current_path").update(f"Selected: {self.selected_path}")
            self.query_one("#current_path_display").update(f"Current: {new_path_obj}")
            
            self.app.log(f"Navigated to: {new_path_obj}")
        except Exception as e:
            self.app.log(f"Tree refresh error: {e}")


class MLGazeConfigApp(App):
    """Main TUI application for MLGaze configuration."""
    
    CSS_PATH = "styles.tcss"
    
    def __init__(self):
        super().__init__()
        self.config = VisualizationConfig()
        self.cameras = []  # Available cameras from session
        self.session_validated = False  # Track if session directory is valid
    
    def show_error(self, message: str) -> None:
        """Show error notification to user."""
        self.notify(message, severity="error", timeout=5)
    
    def show_success(self, message: str) -> None:
        """Show success notification to user."""
        self.notify(message, severity="information", timeout=3)
    
    def show_warning(self, message: str) -> None:
        """Show warning notification to user."""
        self.notify(message, severity="warning", timeout=4)
    
    def validate_directory(self, path: str) -> bool:
        """Validate if directory contains required organized session structure."""
        input_field = self.query_one("#input_directory", Input)
        
        if not path:
            input_field.remove_class("valid-path", "invalid-path", "warning-path")
            return False
            
        path_obj = Path(path)
        
        # Check if path exists
        if not path_obj.exists():
            self.show_error(f"Directory does not exist: {path}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Check if it's actually a directory
        if not path_obj.is_dir():
            self.show_error(f"Path is not a directory: {path}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # Validate organized session structure
        try:
            self._validate_session_structure(path_obj)
            # If validation passes, try to load session metadata to get cameras
            self._load_session_info(path_obj)
        except (FileNotFoundError, ValueError) as e:
            self.show_error(f"Invalid session structure: {e}")
            input_field.remove_class("valid-path", "warning-path")
            input_field.add_class("invalid-path")
            return False
        
        # If we get here, validation passed
        input_field.remove_class("invalid-path", "warning-path")
        input_field.add_class("valid-path")
        self.session_validated = True
        self.show_success("Valid session directory")
        # Now show camera selection UI if multiple cameras available
        self._show_camera_selection_ui()
        return True
    
    def _validate_session_structure(self, session_path: Path) -> None:
        """Validate organized session structure with warnings for missing optional files."""
        # Check required directories
        cameras_dir = session_path / "cameras"
        sensors_dir = session_path / "sensors"
        
        missing_dirs = []
        if not cameras_dir.exists():
            missing_dirs.append("cameras/")
        if not sensors_dir.exists():
            missing_dirs.append("sensors/")
        
        if missing_dirs:
            raise FileNotFoundError(f"Required directories missing: {', '.join(missing_dirs)}")
        
        # Check for at least one camera
        camera_dirs = [d for d in cameras_dir.iterdir() if d.is_dir()]
        if not camera_dirs:
            raise ValueError("No camera directories found in cameras/")
        
        # Check required sensor files
        gaze_data_file = sensors_dir / "gaze_data.csv"
        if not gaze_data_file.exists():
            raise FileNotFoundError("Required file missing: sensors/gaze_data.csv")
        
        # Check optional sensor files
        imu_data_file = sensors_dir / "imu_data.csv"
        if not imu_data_file.exists():
            self.show_warning("Optional file missing: sensors/imu_data.csv (IMU visualization disabled)")
        
        # Validate each camera directory with warnings for optional files
        warnings = []
        for camera_dir in camera_dirs:
            camera_name = camera_dir.name
            
            # Required: frame metadata
            frame_metadata = camera_dir / "frame_metadata.csv"
            if not frame_metadata.exists():
                raise FileNotFoundError(f"Camera {camera_name} missing required frame_metadata.csv")
            
            # Required: at least one frame source
            frames_dir = camera_dir / "frames"
            mlcf_file = camera_dir / "camera_frames.mlcf"
            
            if not frames_dir.exists() and not mlcf_file.exists():
                raise FileNotFoundError(
                    f"Camera {camera_name} missing both frames/ directory and camera_frames.mlcf"
                )
            
            # Optional: gaze screen coordinates
            gaze_screen_coords = camera_dir / "gaze_screen_coords.csv"
            if not gaze_screen_coords.exists():
                warnings.append(f"Camera {camera_name} missing gaze_screen_coords.csv (2D gaze overlay disabled)")
        
        # Show warnings for missing optional files
        if warnings:
            for warning in warnings:
                self.show_warning(warning)
    
    def _load_session_info(self, session_path: Path) -> None:
        """Load session information and update UI for camera selection."""
        try:
            # Try to load metadata for camera info
            metadata_file = session_path / "metadata.json"
            if metadata_file.exists():
                metadata = SessionMetadata.from_json_file(metadata_file)
                self.cameras = metadata.get_camera_names()
            else:
                # Get cameras from directory structure
                cameras_dir = session_path / "cameras"
                self.cameras = [d.name for d in cameras_dir.iterdir() if d.is_dir()]
            
            # Update camera selection UI if multiple cameras
            self._show_camera_selection_ui()
            
        except Exception as e:
            # Continue without camera selection - basic validation passed
            self.cameras = []
    
    def _show_camera_selection_ui(self) -> None:
        """Show camera selection UI after successful directory validation."""
        
        try:
            camera_container = self.query_one("#camera_selection")
            if len(self.cameras) <= 1:
                # Hide camera selection if only one or no cameras
                camera_container.display = False
                # Set the single camera as primary if available
                if len(self.cameras) == 1:
                    self.config.primary_camera = self.cameras[0]
                    self.show_success(f"Single camera found: {self.cameras[0]}")
            else:
                # Show camera selection for multiple cameras
                camera_container.display = True
                
                # Update Select dropdown
                try:
                    camera_select = self.query_one("#primary_camera", Select)
                    
                    # Create options for the Select widget (label, value pairs)
                    options = [(camera_name, camera_name) for camera_name in self.cameras]
                    
                    # Set the options
                    camera_select.set_options(options)
                    
                    # Set first camera as default
                    if self.cameras:
                        camera_select.value = self.cameras[0]
                        self.config.primary_camera = self.cameras[0]
                        self.show_success(f"Multiple cameras found. Primary set to: {self.cameras[0]}")
                        
                except Exception as e:
                    self.show_error(f"Failed to populate camera selection: {e}")
            
            # Also update tracking cameras SelectionList for Object Instance Tracker
            self._update_tracking_cameras_selection()
                    
        except Exception as e:
            self.show_error(f"Camera selection UI error: {e}")
    
    def _hide_camera_selection_ui(self) -> None:
        """Hide camera selection UI when directory becomes invalid."""
        try:
            camera_container = self.query_one("#camera_selection")
            camera_container.display = False
            
            # Also clear tracking cameras selection
            self._update_tracking_cameras_selection(clear=True)
        except Exception as e:
            pass  # Silently handle UI cleanup errors
        self.cameras = []
        self.config.primary_camera = ""
    
    def _update_tracking_cameras_selection(self, clear: bool = False) -> None:
        """Update tracking cameras SelectionList using built-in option methods.
        
        Args:
            clear: If True, clear all options. If False, populate with cameras.
        """
        try:
            # Get the existing SelectionList widget
            selection_list = self.query_one("#instance_tracking_cameras", SelectionList)
            
            # Clear existing options
            selection_list.clear_options()
            
            if not clear and self.cameras:
                # Add camera options (all selected by default)
                camera_options = [(camera_name, camera_name) for camera_name in self.cameras]
                selection_list.add_options(camera_options)
                
                # Select all cameras by default
                for i in range(len(self.cameras)):
                    selection_list.select(i)
                
                # Initialize config with all cameras
                self.config.plugin_configs["ObjectInstanceTracker"]["tracking_cameras"] = "all"
            
        except Exception as e:
            self.show_error(f"Failed to update tracking cameras selection: {e}")
    
    def _update_object_detection_status(self) -> None:
        """Update object detection status indicator."""
        try:
            status_widget = self.query_one("#object_detection_status", Static)
            
            if not self.config.enable_object_detection:
                status_widget.update("Status: Disabled")
                return
            
            if not OBJECT_DETECTION_AVAILABLE:
                status_widget.update("Status: Dependencies missing")
                return
            
            # Check model availability with download capability
            if ObjectDetector:
                custom_path = self.config.object_detection_custom_model_path if self.config.object_detection_model == "custom" else None
                model_status = ObjectDetector.check_model_or_downloadable(self.config.object_detection_model, custom_path)
                
                if self.config.object_detection_model == "custom":
                    if model_status == "local":
                        status_widget.update(f"Status: Ready (custom model loaded, conf: {self.config.object_detection_confidence:.2f})")
                    else:
                        status_widget.update("Status: Custom model not found - check path")
                elif model_status == "local":
                    status_widget.update(f"Status: Ready (local {self.config.object_detection_model} model, conf: {self.config.object_detection_confidence:.2f})")
                elif model_status == "downloadable":
                    status_widget.update(f"Status: Ready (will download {self.config.object_detection_model} model on first use, conf: {self.config.object_detection_confidence:.2f})")
                else:
                    # Check what's available
                    available_local = ObjectDetector.get_available_models()
                    downloadable_models = [size for size in ObjectDetector.MODEL_FILES.keys() 
                                         if size != "custom" and ObjectDetector.check_model_or_downloadable(size) == "downloadable"]
                    
                    if available_local:
                        status_widget.update(f"Status: Model '{self.config.object_detection_model}' unavailable. Local models: {', '.join(available_local)}")
                    elif downloadable_models:
                        status_widget.update(f"Status: Will download on first use. Available: {', '.join(downloadable_models)}")
                    else:
                        status_widget.update("Status: No models available")
            else:
                status_widget.update("Status: ObjectDetector not available")
                
        except Exception as e:
            # Silently handle status update errors
            pass
    
    def _toggle_custom_model_inputs(self, enabled: bool) -> None:
        """Show/hide custom model inputs based on model selection."""
        try:
            # Toggle custom path input
            custom_path_input = self.query_one("#object_detection_custom_path", Input)
            custom_path_input.disabled = not enabled
            
            # Toggle custom classes input
            custom_classes_input = self.query_one("#object_detection_custom_classes", Input)
            custom_classes_input.disabled = not enabled
            
            # Update visual styling
            if enabled:
                custom_path_input.add_class("enabled")
                custom_classes_input.add_class("enabled")
            else:
                custom_path_input.remove_class("enabled")
                custom_classes_input.remove_class("enabled")
                # Clear values when disabled
                custom_path_input.value = ""
                custom_classes_input.value = ""
                self.config.object_detection_custom_model_path = ""
                self.config.object_detection_custom_classes = ""
                
        except Exception as e:
            pass  # Silently handle UI errors
    
    def _create_camera_selection_section(self):
        """Create the camera selection section (initially hidden)."""
        yield Static("Primary Camera (3D View)", classes="section-title", id="camera_section_title")
        with Container(classes="section-content", id="camera_selection") as container:
            container.display = False  # Initially hidden until session is validated
            yield Static("Select primary camera for 3D visualization:", id="camera_instruction")
            yield Select(
                options=[],  # Empty initially, will be populated dynamically
                prompt="Select a camera",
                id="primary_camera"
            )
    
    def _create_spatial_analysis_sections(self):
        """Create the spatial analysis configuration sections."""
        # 3D Heatmap Section
        yield Static("3D Gaze Heatmap", classes="section-title")
        with Container(classes="section-content"):
            yield Checkbox("Enable 3D Heatmap", value=True, id="enable_heatmap")
            yield Checkbox("Show Heatmap Visualization", value=True, id="show_heatmap")
            yield Checkbox("Use Boxes (vs Points)", value=True, id="heatmap_use_boxes")
            
            yield Static("Voxel Size (meters):")
            yield Input(
                placeholder="0.1",
                value="0.1",
                id="heatmap_voxel_size"
            )
            
            yield Static("Minimum Density (points/voxel):")
            yield Input(
                placeholder="5",
                value="5",
                id="heatmap_min_density"
            )
            
            yield Static("Minimum Opacity (0.1-1.0):")
            yield Input(
                placeholder="0.3",
                value="0.3",
                id="heatmap_opacity_min"
            )
            
            yield Static("Maximum Opacity (0.1-1.0):")
            yield Input(
                placeholder="1.0",
                value="1.0",
                id="heatmap_opacity_max"
            )
            
            yield Static("Fill Mode:")
            yield Select(
                options=[
                    ("Solid", "solid"),
                    ("Wireframe", "wireframe")
                ],
                value="solid",
                id="heatmap_fill_mode"
            )
        
        # 3D Clustering Section
        yield Static("3D Gaze Clustering", classes="section-title")
        with Container(classes="section-content"):
            yield Checkbox("Enable 3D Clustering", value=True, id="enable_clustering")
            yield Checkbox("Show Bounding Boxes", value=True, id="clustering_show_bounds")
            
            yield Static("Minimum Cluster Size (points):")
            yield Input(
                placeholder="30",
                value="30",
                id="clustering_min_size"
            )
            
            yield Static("Minimum Samples:")
            yield Input(
                placeholder="5",
                value="5",
                id="clustering_min_samples"
            )
            
            yield Static("Epsilon Distance (meters):")
            yield Input(
                placeholder="0.15",
                value="0.15",
                id="clustering_epsilon"
            )
            
            yield Static("Point Opacity (0.1-1.0):")
            yield Input(
                placeholder="0.8",
                value="0.8",
                id="clustering_point_opacity"
            )
            
            yield Static("Bounding Box Opacity (0.1-1.0):")
            yield Input(
                placeholder="0.3",
                value="0.3",
                id="clustering_bound_opacity"
            )
        
        # Object Instance Tracker Section
        yield Static("Object Instance Tracker", classes="section-title")
        with Container(classes="section-content"):
            yield Checkbox("Enable Object Instance Tracker", value=True, id="enable_instance_tracker")
            yield Checkbox("Show 3D Bounding Boxes", value=True, id="instance_show_3d_boxes")
            yield Checkbox("Generate CSV Reports", value=True, id="instance_generate_reports")
            
            yield Static("Tracking Cameras:")
            with Container(classes="selection-container", id="instance_tracking_cameras_container"):
                yield Static("Select cameras to track objects from:", id="tracking_cameras_instruction")
                yield SelectionList(id="instance_tracking_cameras")
            
            yield Static("Lifecycle Mode:")
            yield RadioSet(
                "persistent", "temporal", "visit-based",
                id="instance_lifecycle_mode"
            )
            
            yield Static("IoU Threshold (0.1-0.9):")
            yield Input(
                placeholder="0.3",
                value="0.3",
                id="instance_iou_threshold"
            )
            
            yield Static("Max Frame Gap (1-30):")
            yield Input(
                placeholder="10",
                value="10",
                id="instance_max_frame_gap"
            )
            
            yield Static("Min Confidence (0.1-1.0):")
            yield Input(
                placeholder="0.5",
                value="0.5",
                id="instance_min_confidence"
            )
            
            yield Static("Min Gaze Points for 3D Bbox (10-200):")
            yield Input(
                placeholder="50",
                value="50",
                id="instance_min_gaze_points"
            )
            
            yield Static("Gaze Time Window (ms) (50-500):")
            yield Input(
                placeholder="100",
                value="100",
                id="instance_time_window"
            )
            
            yield Static("3D Box Opacity (0.1-1.0):")
            yield Input(
                placeholder="0.4",
                value="0.4",
                id="instance_box_opacity"
            )

        # Task Segmentation Section
        yield Static("Task Segmentation", classes="section-title")
        with Container(classes="section-content"):
            yield Checkbox("Enable Task Segmentation", value=True, id="enable_task_segmentation")
            yield Static("Task segmentation parameters are configured in task_segmenter section of config.yaml")

    def _create_gaze_state_filter_section(self):
        """Create the gaze state filter configuration section."""
        yield Static("Gaze State Filtering", classes="section-title")
        with Container(classes="section-content"):
            # Enable filtering checkbox
            yield Checkbox("Enable Gaze State Filtering", value=False, id="enable_gaze_filter")
            
            # Multi-select list for gaze states
            yield Static("Select gaze states to include (when enabled):")
            
            # Available gaze states from the data
            gaze_states = [
                ("Fixation", "Fixation"),
                ("Pursuit", "Pursuit"),
                ("Saccade", "Saccade"),
                ("Blink", "Blink"),
                ("Unknown", "Unknown"),
                ("Wink Left", "WinkLeft"),
                ("Wink Right", "WinkRight")
            ]
            
            # Create SelectionList with all states initially selected
            yield SelectionList(
                *gaze_states,
                id="gaze_states_selection"
            )
            
            yield Static("Note: Empty selection means all states will be included", classes="help-text")
    
    def _create_object_detection_section(self):
        """Create the object detection configuration section."""
        yield Static("Object Detection", classes="section-title")
        with Container(classes="section-content"):
            # Enable/Disable checkbox
            yield Checkbox("Enable Object Detection", value=False, id="enable_object_detection")
            
            if OBJECT_DETECTION_AVAILABLE and ObjectDetector:
                # Get available models
                available_models = ObjectDetector.get_available_models()
                if not available_models:
                    available_models = ["base"]  # Default fallback
                
                # Model selection
                yield Static("Model Size:")
                model_options = [
                    ("Nano (Fastest)", "nano"),
                    ("Small", "small"),
                    ("Medium", "medium"),
                    ("Base (Most Accurate)", "base"),
                    ("Custom Fine-tuned", "custom")
                ]
                yield Select(
                    options=model_options,
                    value="base",
                    id="object_detection_model"
                )
                
                # Custom model path (initially hidden)
                yield Static("Custom Model Path:", id="custom_model_label")
                yield Input(
                    placeholder="/path/to/custom-model.pth",
                    value="",
                    disabled=True,
                    id="object_detection_custom_path"
                )
                
                # Custom class names (initially hidden)
                yield Static("Custom Classes (comma-separated):", id="custom_classes_label")
                yield Input(
                    placeholder="person,car,bike,dog,cat",
                    value="",
                    disabled=True,
                    id="object_detection_custom_classes"
                )
                
                # Confidence threshold
                yield Static("Confidence Threshold (0.0 - 1.0):")
                yield Input(
                    placeholder="0.5",
                    value="0.5",
                    id="object_detection_confidence"
                )
                
                # NMS threshold
                yield Static("NMS Threshold (0.0 - 1.0):")
                yield Input(
                    placeholder="0.5",
                    value="0.5",
                    id="object_detection_nms_threshold"
                )
                
                # Target classes filter
                yield Static("Target Classes (comma-separated, empty = all):")
                yield Input(
                    placeholder="person,car,bike",
                    value="",
                    id="object_detection_target_classes"
                )
                
                # Device selection
                yield Static("Processing Device:")
                device_options = [
                    ("Auto (GPU if available)", "auto"),
                    ("CPU Only", "cpu"),
                    ("GPU (CUDA)", "cuda"),
                    ("Apple Silicon (MPS)", "mps")
                ]
                yield Select(
                    options=device_options,
                    value="auto",
                    id="object_detection_device"
                )
                
                # Image Preprocessing Mode
                yield Static("Preprocessing Mode:")
                preprocessing_options = [
                    ("Center Crop (Recommended)", "center_crop"),
                    ("Padding (Preserve edges)", "padding"),
                    ("None (Direct resize)", "none")
                ]
                yield Select(
                    options=preprocessing_options,
                    value="center_crop",
                    id="object_detection_preprocessing_mode",
                    tooltip="Center crop: Better accuracy, may lose edges. Padding: Keeps full image, may reduce accuracy"
                )
                
                # Status indicator
                yield Static("", id="object_detection_status")
                
            else:
                yield Static("Object detection not available. Install required dependencies.", id="object_detection_error")
    
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        
        with ScrollableContainer():
            yield Static("MLGaze Viewer Configuration", classes="section-title")
            
            with Container(classes="grid-container"):
                # Left Column - File and Temporal Settings  
                with Container(classes="left-column"):
                    yield Static("Data Files", classes="section-title")
                    with Container(classes="section-content"):
                        yield Static("Input Directory:")
                        with Horizontal():
                            yield Input(
                                placeholder="Select session directory...", 
                                value="", 
                                id="input_directory"
                            )
                            yield Button("Browse", id="browse_directory")
                    
                    # Camera Selection Section (dynamic based on available cameras)
                    yield from self._create_camera_selection_section()
                    
                    yield Static("Timestamp Synchronization", classes="section-title")
                    with Container(classes="section-content"):
                        yield Static("Choose how to handle sensors with different start/stop times:")
                        yield RadioSet(
                            RadioButton("Union: All data (first start to last stop)", value=True, id="union_mode"),
                            RadioButton("Intersection: Synchronized data only (latest start to earliest stop)", id="intersection_mode"),
                            id="timestamp_sync_mode"
                        )
                    
                    yield Static("Trail Effects", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Enable Trail Fade", value=True, id="fade_trail")
                        
                        yield Static("Fade Duration (seconds):")
                        yield Input(
                            placeholder="5.0", 
                            value="5.0", 
                            id="fade_duration"
                        )
                    
                    yield Static("Sliding Window (3D only)", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Enable Sliding Window", value=False, id="sliding_window")
                        
                        yield Static("Window Duration (seconds):")
                        yield Input(
                            placeholder="10.0", 
                            value="10.0", 
                            id="window_duration"
                        )
                        
                        yield Static("Update Rate (seconds):")
                        yield Input(
                            placeholder="0.5", 
                            value="0.5", 
                            id="update_rate"
                        )
                        
                        yield Checkbox("Apply to 3D Gaze Points", value=True, id="window_gaze")
                        yield Checkbox("Apply to 3D Trajectories", value=True, id="window_trajectory")
                        yield Checkbox("Apply to Camera Position", value=True, id="window_camera")
                
                # Right Column - Visualization Settings
                with Container(classes="right-column"):
                    yield Static("Visualization Options", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Show Point Cloud", value=True, id="point_cloud")
                        yield Checkbox("Show Gaze Trajectory", value=True, id="gaze_trajectory")
                        yield Checkbox("Show Camera Path", value=True, id="camera_path")
                        yield Checkbox("Color by Gaze State", value=True, id="color_by_state")
                        yield Checkbox("Test Y-Flip", value=False, id="y_flip")
                        yield Checkbox("Show Coordinate Indicators", value=True, id="coord_indicators")
                    
                    # Object Detection Section
                    yield from self._create_object_detection_section()
                    
                    # Spatial Analysis Sections
                    yield from self._create_spatial_analysis_sections()
                    
                    # Gaze State Filter Section
                    yield from self._create_gaze_state_filter_section()
                    
                    yield Static("IMU Sensor Data", classes="section-title")
                    with Container(classes="section-content"):
                        yield Checkbox("Show IMU Data", value=True, id="show_imu")
            
            # Action Buttons (full width, outside grid)
            with Container(classes="button-container"):
                yield Button("Start Visualization", variant="primary", id="start_viz")
                yield Button("Reset to Defaults", variant="default", id="reset")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the form and set focus."""
        self.title = "MLGaze Viewer"
        self.sub_title = "Configure Visualization Parameters"
        
        # Set focus to first input field
        self.query_one("#input_directory", Input).focus()
        
        # Update object detection status on startup
        self._update_object_detection_status()
    
    
    @on(Select.Changed)
    def selection_changed(self, event: Select.Changed) -> None:
        """Handle all select widget changes."""
        select_id = event.select.id
        selected_value = event.value
        
        if select_id == "primary_camera":
            if selected_value:  # Ensure a value was selected
                self.config.primary_camera = selected_value
        elif select_id == "object_detection_model":
            if selected_value:
                self.config.object_detection_model = selected_value
                # Toggle custom model inputs
                self._toggle_custom_model_inputs(selected_value == "custom")
                self._update_object_detection_status()
        elif select_id == "object_detection_device":
            if selected_value:
                self.config.object_detection_device = selected_value
                self._update_object_detection_status()
        elif select_id == "object_detection_preprocessing_mode":
            if selected_value:
                self.config.object_detection_preprocessing_mode = selected_value
        elif select_id == "heatmap_fill_mode":
            if selected_value:
                self.config.plugin_configs["Gaze3DHeatmap"]["fill_mode"] = selected_value
    
    @on(Checkbox.Changed)
    def checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id
        value = event.value
        
        if checkbox_id == "fade_trail":
            self.config.enable_fade_trail = value
        elif checkbox_id == "point_cloud":
            self.config.show_point_cloud = value
        elif checkbox_id == "gaze_trajectory":
            self.config.show_gaze_trajectory = value
        elif checkbox_id == "camera_path":
            self.config.show_camera_trajectory = value
        elif checkbox_id == "color_by_state":
            self.config.color_by_gaze_state = value
        elif checkbox_id == "y_flip":
            self.config.test_y_flip = value
        elif checkbox_id == "coord_indicators":
            self.config.show_coordinate_indicators = value
        elif checkbox_id == "sliding_window":
            self.config.enable_sliding_window = value
        elif checkbox_id == "window_gaze":
            self.config.sliding_window_3d_gaze = value
        elif checkbox_id == "window_trajectory":
            self.config.sliding_window_3d_trajectory = value
        elif checkbox_id == "window_camera":
            self.config.sliding_window_camera = value
        elif checkbox_id == "show_imu":
            self.config.show_imu_data = value
        elif checkbox_id == "enable_object_detection":
            self.config.enable_object_detection = value
            self._update_object_detection_status()
        # Spatial Analysis checkboxes
        elif checkbox_id == "enable_heatmap":
            if "Gaze3DHeatmap" in self.config.enabled_plugins and not value:
                self.config.enabled_plugins.remove("Gaze3DHeatmap")
            elif "Gaze3DHeatmap" not in self.config.enabled_plugins and value:
                self.config.enabled_plugins.append("Gaze3DHeatmap")
        elif checkbox_id == "show_heatmap":
            self.config.plugin_configs["Gaze3DHeatmap"]["show_heatmap"] = value
        elif checkbox_id == "heatmap_use_boxes":
            self.config.plugin_configs["Gaze3DHeatmap"]["use_boxes"] = value
        elif checkbox_id == "enable_clustering":
            if "Gaze3DClustering" in self.config.enabled_plugins and not value:
                self.config.enabled_plugins.remove("Gaze3DClustering")
            elif "Gaze3DClustering" not in self.config.enabled_plugins and value:
                self.config.enabled_plugins.append("Gaze3DClustering")
        elif checkbox_id == "clustering_show_bounds":
            self.config.plugin_configs["Gaze3DClustering"]["show_bounds"] = value
        elif checkbox_id == "enable_instance_tracker":
            if "ObjectInstanceTracker" in self.config.enabled_plugins and not value:
                self.config.enabled_plugins.remove("ObjectInstanceTracker")
            elif "ObjectInstanceTracker" not in self.config.enabled_plugins and value:
                self.config.enabled_plugins.append("ObjectInstanceTracker")
        elif checkbox_id == "instance_show_3d_boxes":
            self.config.plugin_configs["ObjectInstanceTracker"]["show_3d_boxes"] = value
        elif checkbox_id == "instance_generate_reports":
            self.config.plugin_configs["ObjectInstanceTracker"]["generate_reports"] = value
        elif checkbox_id == "enable_task_segmentation":
            if "TaskSegmentationPlugin" in self.config.enabled_plugins and not value:
                self.config.enabled_plugins.remove("TaskSegmentationPlugin")
            elif "TaskSegmentationPlugin" not in self.config.enabled_plugins and value:
                self.config.enabled_plugins.append("TaskSegmentationPlugin")
        elif checkbox_id == "enable_gaze_filter":
            # Apply filter enabled state to spatial plugins
            self.config.plugin_configs["Gaze3DHeatmap"]["filter_enabled"] = value
            self.config.plugin_configs["Gaze3DClustering"]["filter_enabled"] = value
    
    @on(SelectionList.SelectedChanged)
    def selection_list_changed(self, event: SelectionList.SelectedChanged) -> None:
        """Handle selection list changes."""
        if event.selection_list.id == "gaze_states_selection":
            # Get the selected gaze states
            selected_states = list(event.selection_list.selected)
            
            # Apply to both spatial plugins
            self.config.plugin_configs["Gaze3DHeatmap"]["gaze_states_filter"] = selected_states
            self.config.plugin_configs["Gaze3DClustering"]["gaze_states_filter"] = selected_states
        elif event.selection_list.id == "instance_tracking_cameras":
            # Get the selected cameras for object tracking
            selected_cameras = list(event.selection_list.selected)
            
            # Set tracking cameras config - use 'all' if nothing selected or all selected
            if not selected_cameras or len(selected_cameras) == len(self.cameras):
                self.config.plugin_configs["ObjectInstanceTracker"]["tracking_cameras"] = "all"
            else:
                self.config.plugin_configs["ObjectInstanceTracker"]["tracking_cameras"] = selected_cameras
    
    @on(RadioSet.Changed)
    def radioset_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button changes."""
        if event.radio_set.id == "timestamp_sync_mode":
            selected_button = event.pressed
            if selected_button.id == "union_mode":
                self.config.timestamp_sync_mode = "union"
            elif selected_button.id == "intersection_mode":
                self.config.timestamp_sync_mode = "intersection"
        elif event.radio_set.id == "instance_lifecycle_mode":
            # Get the selected lifecycle mode
            selected_button = event.pressed
            if selected_button:
                self.config.plugin_configs["ObjectInstanceTracker"]["lifecycle_mode"] = selected_button.label
    
    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        """Handle input field changes."""
        input_id = event.input.id
        value = event.value
        
        try:
            if input_id == "input_directory":
                self.config.input_directory = value if value else ""
                # Reset session validation state when directory changes
                if not self.session_validated or value != self.config.input_directory:
                    self.session_validated = False
                    self._hide_camera_selection_ui()
                
                # Validate directory path with user feedback
                if value and len(value) > 3:  # Avoid validating every keystroke
                    self.validate_directory(value)
                elif not value:
                    # Clear validation when input is empty
                    input_field = self.query_one("#input_directory", Input)
                    input_field.remove_class("valid-path", "invalid-path", "warning-path")
                    
            elif input_id == "fade_duration":
                if value:
                    duration = float(value)
                    if duration < 0.1:
                        self.show_warning("Fade duration must be at least 0.1 seconds")
                    elif duration > 60:
                        self.show_warning("Fade duration is very long")
                    else:
                        self.config.fade_duration = duration
                else:
                    self.config.fade_duration = 5.0
                    
            elif input_id == "window_duration":
                if value:
                    duration = float(value)
                    if duration < 1.0:
                        self.show_warning("Window duration must be at least 1 second")
                    elif duration > 300:
                        self.show_warning("Window duration is very long (>5 minutes)")
                    else:
                        self.config.sliding_window_duration = duration
                else:
                    self.config.sliding_window_duration = 10.0
                    
            elif input_id == "update_rate":
                if value:
                    rate = float(value)
                    if rate < 0.1:
                        self.show_warning("Update rate must be at least 0.1 seconds")
                    elif rate > 10:
                        self.show_warning("Update rate is very slow (>10 seconds)")
                    else:
                        self.config.sliding_window_update_rate = rate
                else:
                    self.config.sliding_window_update_rate = 0.5
                    
            elif input_id == "object_detection_confidence":
                if value:
                    confidence = float(value)
                    if confidence < 0.0:
                        self.show_warning("Confidence must be at least 0.0")
                    elif confidence > 1.0:
                        self.show_warning("Confidence must be at most 1.0")
                    else:
                        self.config.object_detection_confidence = confidence
                        self._update_object_detection_status()
                else:
                    self.config.object_detection_confidence = 0.5
                    
            elif input_id == "object_detection_nms_threshold":
                if value:
                    nms_threshold = float(value)
                    if nms_threshold < 0.0:
                        self.show_warning("NMS threshold must be at least 0.0")
                    elif nms_threshold > 1.0:
                        self.show_warning("NMS threshold must be at most 1.0")
                    else:
                        self.config.object_detection_nms_threshold = nms_threshold
                else:
                    self.config.object_detection_nms_threshold = 0.5
                    
            elif input_id == "object_detection_custom_path":
                self.config.object_detection_custom_model_path = value
                self._update_object_detection_status()
                
            elif input_id == "object_detection_custom_classes":
                self.config.object_detection_custom_classes = value
                
            elif input_id == "object_detection_target_classes":
                self.config.object_detection_target_classes = value
                
            # Spatial Analysis Parameters - Heatmap
            elif input_id == "heatmap_voxel_size":
                if value:
                    voxel_size = float(value)
                    if 0.05 <= voxel_size <= 0.5:
                        self.config.plugin_configs["Gaze3DHeatmap"]["voxel_size"] = voxel_size
                        
            elif input_id == "heatmap_min_density":
                if value:
                    min_density = int(value)
                    if 1 <= min_density <= 50:
                        self.config.plugin_configs["Gaze3DHeatmap"]["min_density"] = min_density
                        
            elif input_id == "heatmap_opacity_min":
                if value:
                    opacity = float(value)
                    if 0.1 <= opacity <= 1.0:
                        self.config.plugin_configs["Gaze3DHeatmap"]["opacity_min"] = opacity
                        
            elif input_id == "heatmap_opacity_max":
                if value:
                    opacity = float(value)
                    if 0.1 <= opacity <= 1.0:
                        self.config.plugin_configs["Gaze3DHeatmap"]["opacity_max"] = opacity
                        
            # Spatial Analysis Parameters - Clustering
            elif input_id == "clustering_min_size":
                if value:
                    min_size = int(value)
                    if 10 <= min_size <= 500:
                        self.config.plugin_configs["Gaze3DClustering"]["min_cluster_size"] = min_size
                        
            elif input_id == "clustering_min_samples":
                if value:
                    min_samples = int(value)
                    if 1 <= min_samples <= 50:
                        self.config.plugin_configs["Gaze3DClustering"]["min_samples"] = min_samples
                        
            elif input_id == "clustering_epsilon":
                if value:
                    epsilon = float(value)
                    if 0.05 <= epsilon <= 1.0:
                        self.config.plugin_configs["Gaze3DClustering"]["epsilon_m"] = epsilon
                        
            elif input_id == "clustering_point_opacity":
                if value:
                    opacity = float(value)
                    if 0.1 <= opacity <= 1.0:
                        self.config.plugin_configs["Gaze3DClustering"]["point_opacity"] = opacity
                        
            elif input_id == "clustering_bound_opacity":
                if value:
                    opacity = float(value)
                    if 0.1 <= opacity <= 1.0:
                        self.config.plugin_configs["Gaze3DClustering"]["bound_opacity"] = opacity
            
            # Object Instance Tracker Parameters
            elif input_id == "instance_iou_threshold":
                if value:
                    iou_threshold = float(value)
                    if 0.1 <= iou_threshold <= 0.9:
                        self.config.plugin_configs["ObjectInstanceTracker"]["iou_threshold"] = iou_threshold
                        
            elif input_id == "instance_max_frame_gap":
                if value:
                    max_frame_gap = int(value)
                    if 1 <= max_frame_gap <= 30:
                        self.config.plugin_configs["ObjectInstanceTracker"]["max_frame_gap"] = max_frame_gap
                        
            elif input_id == "instance_min_confidence":
                if value:
                    min_confidence = float(value)
                    if 0.1 <= min_confidence <= 1.0:
                        self.config.plugin_configs["ObjectInstanceTracker"]["min_confidence"] = min_confidence
                        
            elif input_id == "instance_min_gaze_points":
                if value:
                    min_gaze_points = int(value)
                    if 10 <= min_gaze_points <= 200:
                        self.config.plugin_configs["ObjectInstanceTracker"]["min_gaze_points"] = min_gaze_points
                        
            elif input_id == "instance_time_window":
                if value:
                    time_window = int(value)
                    if 50 <= time_window <= 500:
                        self.config.plugin_configs["ObjectInstanceTracker"]["gaze_time_window_ms"] = time_window
                        
            elif input_id == "instance_box_opacity":
                if value:
                    box_opacity = float(value)
                    if 0.1 <= box_opacity <= 1.0:
                        self.config.plugin_configs["ObjectInstanceTracker"]["box_opacity"] = box_opacity
                    
        except ValueError as e:
            if "fade_duration" in input_id:
                self.show_error("Value must be a number")
            else:
                self.show_error(f"Invalid input: {e}")
    
    @on(Button.Pressed, "#start_viz")
    def start_visualization(self) -> None:
        """Handle Start Visualization button."""
        if not self.session_validated or not self.config.input_directory:
            self.show_error("Please select a valid session directory first")
            return
        
        # Sync object detection configuration
        self.config.sync_object_detection_config()
        
        self.exit(self.config)
    
    @on(Button.Pressed, "#reset")
    def reset_to_defaults(self) -> None:
        """Handle Reset button."""
        self.config = VisualizationConfig()
        
        # Reset session validation
        self.session_validated = False
        self._hide_camera_selection_ui()
        
        # Update UI elements with default values
        self.query_one("#input_directory", Input).value = ""
        self.query_one("#fade_duration", Input).value = "5.0"
        self.query_one("#window_duration", Input).value = "10.0"
        self.query_one("#update_rate", Input).value = "0.5"
        
        # Reset checkboxes
        self.query_one("#fade_trail", Checkbox).value = True
        self.query_one("#point_cloud", Checkbox).value = True
        self.query_one("#gaze_trajectory", Checkbox).value = True
        self.query_one("#camera_path", Checkbox).value = True
        self.query_one("#color_by_state", Checkbox).value = True
        self.query_one("#y_flip", Checkbox).value = False
        self.query_one("#coord_indicators", Checkbox).value = True
        self.query_one("#sliding_window", Checkbox).value = False
        self.query_one("#window_gaze", Checkbox).value = True
        self.query_one("#window_trajectory", Checkbox).value = True
        self.query_one("#window_camera", Checkbox).value = True
        self.query_one("#show_imu", Checkbox).value = True
        
        # Reset radio buttons
        self.query_one("#union_mode", RadioButton).value = True
        self.query_one("#intersection_mode", RadioButton).value = False
        
        # Reset object detection settings
        try:
            self.query_one("#enable_object_detection", Checkbox).value = False
            self.query_one("#object_detection_model", Select).value = "base"
            self.query_one("#object_detection_confidence", Input).value = "0.5"
            self.query_one("#object_detection_nms_threshold", Input).value = "0.5"
            self.query_one("#object_detection_device", Select).value = "auto"
            self.query_one("#object_detection_preprocessing_mode", Select).value = "center_crop"
            self.query_one("#object_detection_custom_path", Input).value = ""
            self.query_one("#object_detection_custom_classes", Input).value = ""
            self.query_one("#object_detection_target_classes", Input).value = ""
            self._toggle_custom_model_inputs(False)
            self._update_object_detection_status()
        except Exception:
            pass  # Silently handle if object detection widgets don't exist
    
    @on(Button.Pressed, "#browse_directory")
    def browse_directory(self) -> None:
        """Open directory browser and update input field with selected path."""
        try:
            current_path = self.query_one("#input_directory", Input).value or "."
            browser = DirectoryBrowserScreen(current_path)
            
            # Push screen with callback instead of waiting
            self.push_screen(browser, callback=self.on_directory_selected)
        except Exception as e:
            self.show_error(f"Directory browser error: {e}")
    
    def on_directory_selected(self, result: str | None) -> None:
        """Handle directory selection result from browser."""
        if result:  # User selected a directory
            try:
                input_field = self.query_one("#input_directory", Input)
                input_field.value = result
                self.config.input_directory = result
                # Validate the selected directory
                self.validate_directory(result)
            except Exception as e:
                self.show_error(f"Error updating directory selection: {e}")


def run_configuration_tui() -> Optional[VisualizationConfig]:
    """Run the TUI configuration app and return the configuration.
    
    Returns:
        VisualizationConfig if user clicked "Start Visualization", None if cancelled
    """
    app = MLGazeConfigApp()
    result = app.run()
    
    # If result is a VisualizationConfig, user clicked start
    if isinstance(result, VisualizationConfig):
        return result
    else:
        # User cancelled or closed app
        return None


if __name__ == "__main__":
    config = run_configuration_tui()
    if config:
        print("Configuration created:")
        print(f"  Input directory: {config.input_directory}")
        print(f"  Fade trail: {'enabled' if config.enable_fade_trail else 'disabled'}")
        print(f"  Show point cloud: {config.show_point_cloud}")
        print(f"  Show trajectories: {config.show_gaze_trajectory}")
    else:
        print("Configuration cancelled")