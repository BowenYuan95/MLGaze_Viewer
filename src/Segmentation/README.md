# Task-Based Session Segmentation

**Version**: 2.0
**Status**: Production Ready
**Test Coverage**: 34 tests passing

## Overview

This module implements task-based session segmentation for MLGaze Viewer, identifying spatially stable and visually focused periods within Magic Leap 2 recording sessions using a configurable sliding window approach with percentage-based tolerance.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Segmentation Modes](#segmentation-modes)
- [Configuration](#configuration)
- [Algorithm Flow](#algorithm-flow)
- [Data Structures](#data-structures)
- [Usage Examples](#usage-examples)
- [Export Formats](#export-formats)
- [Testing](#testing)
- [File Structure](#file-structure)

---

## Features

### Core Capabilities

- **Dual Stability Modes**:
  - **Percentage Mode** (default): Allows configurable tolerance for unstable samples (20% threshold)
  - **Strict Mode**: All samples must meet stability criteria (original behavior)

- **Multi-Criteria Stability Detection**:
  - Spatial stability (horizontal X-Z plane, ignoring vertical Y)
  - Gaze deviation tracking between consecutive samples
  - Focus distance validation
  - Hand tracking validation (optional)

- **Advanced Processing**:
  - 1.5-second sliding window with overlapping analysis
  - Automatic gaze state filtering (removes Blink/Unknown states)
  - Overlap-based segment merging
  - Time buffer application (configurable start/end buffers)
  - Location consolidation detection (<0.5m movement)

- **Flexible Configuration**:
  - All stability criteria are optional and independently configurable
  - Runtime parameter adjustment via YAML configuration
  - Automatic fallback to strict mode for small sample counts

---

## Quick Start

### Basic Usage

```python
from src.utils import DataLoader
from src.Segmentation import TaskSegmenter, SegmentationConfig, export_task_segments

# 1. Load session data
loader = DataLoader(verbose=True)
session = loader.load_session("path/to/session")

# 2. Create configuration (uses percentage mode by default)
config = SegmentationConfig()

# 3. Run segmentation
segmenter = TaskSegmenter(config)
results = segmenter.segment(session)

# 4. Export results
export_task_segments(
    results,
    output_dir=session.input_directory,
    formats=['csv', 'json', 'summary']
)

print(f"Detected {len(results)} task segments")
for segment in results:
    print(f"  {segment.segmentation_id}: {segment.duration_s:.1f}s")
```

### Custom Configuration

```python
config = SegmentationConfig(
    # Mode selection
    stability_mode="percentage",  # or "strict"

    # Percentage thresholds (only used in percentage mode)
    head_instability_threshold_percent=20.0,  # Allow 20% unstable samples
    gaze_instability_threshold_percent=20.0,
    hand_untracked_threshold_percent=20.0,

    # Optional criteria
    require_spatial_stability=True,
    require_gaze_stability=True,
    require_hand_tracking=False,

    # Thresholds (used by both modes)
    spatial_stability_radius_m=0.3,
    gaze_deviation_threshold_deg=20.0,
    stability_time_window_s=1.5
)
```

---

## Segmentation Modes

### Percentage Mode (Default)

**Concept**: Allows a configurable percentage of unstable samples within each sliding window while still marking the window as stable.

**How It Works**:
- For each 1.5s window, calculates the percentage of unstable samples
- Window is stable if: `unstable_percentage ≤ threshold` (default 20%)
- Tolerates noise, sensor dropouts, and brief instabilities
- More robust for real-world data with inherent sensor noise

**When to Use**:
- Default choice for most use cases
- When dealing with noisy sensor data
- When brief tracking losses are acceptable
- When you want more lenient segmentation

**Example**:
```python
# In a 5-sample window with 20% threshold:
# - 1 unstable sample = 20% → STABLE (exactly at threshold)
# - 2 unstable samples = 40% → UNSTABLE (exceeds threshold)
```

### Strict Mode

**Concept**: All samples within a window must meet stability criteria.

**How It Works**:
- For each 1.5s window, checks every single sample
- Window is unstable if ANY sample fails ANY enabled criterion
- Zero tolerance for instability
- Original implementation behavior

**When to Use**:
- When you need very high confidence segments
- When data quality is consistently high
- When any instability should break a segment
- For conservative task detection

**Switching Modes**:
```python
# Strict mode
config = SegmentationConfig(stability_mode="strict")

# Percentage mode
config = SegmentationConfig(stability_mode="percentage")
```

### Automatic Fallback

The system automatically falls back to strict mode when a window has fewer than `min_samples_for_percentage_mode` samples (default: 5), as percentage calculations become unreliable with very small sample counts.

---

## Configuration

### Configuration Parameters

```yaml
task_segmenter:
  # === Mode Selection ===
  stability_mode: "percentage"  # "strict" or "percentage"

  # === Optional Criteria ===
  require_spatial_stability: true
  require_gaze_stability: true
  require_hand_tracking: false

  # === Spatial Stability ===
  spatial_stability_radius_m: 0.3           # Horizontal (X-Z) distance threshold
  spatial_stability_min_duration_s: 1.0
  stability_time_window_s: 1.5              # Sliding window size
  stability_buffer_start_s: 0.5             # Pre-segment buffer
  stability_buffer_end_s: 0.5               # Post-segment buffer

  # === Gaze Analysis ===
  gaze_deviation_threshold_deg: 20.0        # Max angle between consecutive samples
  focus_distance_threshold_m: 1.5           # Max distance from head to gaze origin
  gaze_smoothing_window_s: 1.0              # EMA window (currently unused)
  gaze_sampling_interval_s: 0.5             # Sampling rate (currently unused)
  filtered_gaze_states: ["Blink", "Unknown"]  # States to exclude

  # === Signal Processing (Currently Unused) ===
  head_filter_alpha: 0.48
  outlier_displacement_threshold_m: 1.5
  outlier_time_window_ms: 100.0
  outlier_speed_threshold_ms: 3.0

  # === Segment Processing ===
  merge_consecutive_gap_s: 1.5              # Gap threshold for grouping stable timestamps
  location_consolidation_threshold_m: 0.5   # Mark as consolidated if movement < 0.5m
  min_segment_duration_s: 1.0               # Minimum segment length

  # === Percentage Mode Parameters ===
  head_instability_threshold_percent: 20.0  # Allow 20% unstable head positions
  gaze_instability_threshold_percent: 20.0  # Allow 20% unstable gaze samples
  hand_untracked_threshold_percent: 20.0    # Allow 20% untracked hand timestamps
  min_samples_for_percentage_mode: 5        # Fallback to strict if fewer samples
```

### Parameter Descriptions

#### Stability Detection
- **spatial_stability_radius_m**: Maximum horizontal (X-Z plane) distance from first position in window. **Y (vertical) component is ignored** to prevent head bobbing from breaking stability.
- **stability_time_window_s**: Duration of sliding window. Larger windows = stricter stability requirements.
- **gaze_deviation_threshold_deg**: Maximum angle between consecutive gaze direction vectors.

#### Percentage Thresholds
- **head_instability_threshold_percent**: Maximum percentage of positions outside stability radius (0-100). Lower = stricter.
- **gaze_instability_threshold_percent**: Maximum percentage of gaze pairs exceeding deviation threshold (0-100).
- **hand_untracked_threshold_percent**: Maximum percentage of timestamps with no tracked hands (0-100).

#### Segment Processing
- **merge_consecutive_gap_s**: Stable timestamps separated by ≤ this gap are grouped together.
- **stability_buffer_start_s/end_s**: Time added before/after detected segments to capture transitions.
- **location_consolidation_threshold_m**: If start/end positions are within this distance, mark segment as consolidated.

---

## Algorithm Flow

### Overview

1. **Validate Session Data** - Check for required camera and sensor data
2. **Extract Camera Positions** - Get 3D head positions from primary camera
3. **Filter Gaze Data** - Remove Blink/Unknown states
4. **Extract Hand Tracking** (optional) - Load hand tracking data if enabled
5. **Sliding Window Detection** - Apply mode-specific stability checks
6. **Group Stable Timestamps** - Merge consecutive stable periods (≤1.5s gap)
7. **Apply Time Buffers** - Extend boundaries by configured amounts
8. **Merge Overlapping Segments** - Combine segments that overlap/touch
9. **Consolidate Locations** - Mark segments with <0.5m movement
10. **Generate Results** - Create TaskSegmentResult objects with IDs

### Detailed Sliding Window Process

For each timestamp in the dataset:

#### 1. Define Window
```
Current Timestamp: T
Window Range: [T, T + 1.5s]
Collect all samples within this range
```

#### 2. Mode Selection
```
IF window_samples < min_samples_for_percentage_mode (5):
    Use strict mode (regardless of config)
ELSE IF stability_mode == "percentage":
    Use percentage mode
ELSE:
    Use strict mode
```

#### 3. Percentage Mode Checks
```python
# Calculate instability percentages
head_unstable_pct = count(positions_outside_radius) / total_positions * 100
gaze_unstable_pct = count(deviation_pairs_>20°) / total_pairs * 100
hand_untracked_pct = count(timestamps_no_hands) / total_timestamps * 100

# Window is stable if ALL enabled criteria pass:
is_stable = (
    head_unstable_pct <= 20.0 AND
    gaze_unstable_pct <= 20.0 AND
    hand_untracked_pct <= 20.0
)
```

#### 4. Strict Mode Checks
```python
# Window is stable if ALL samples in ALL enabled criteria pass:
is_stable = (
    all_positions_within_radius AND
    all_gaze_pairs_≤20° AND
    all_timestamps_have_hands
)
```

#### 5. Mark Timestamp
If `is_stable == True`, add current timestamp to `stable_timestamps` list.

### Reference Points

- **Head Position**: Distance calculated from **FIRST position** in window (matches current implementation)
- **Gaze Direction**: Deviation calculated between **CONSECUTIVE samples** (pairs: [0,1], [1,2], [2,3]...)
- **Distance Calculation**: Only **X and Z** components used (horizontal plane), **Y ignored** (vertical movement)

### Merging Logic

**After time buffering, segments are merged if they overlap or touch:**

```python
# Only merge if next_start <= current_end
if next_segment.start <= current_segment.end:
    merged_segment = (
        current_segment.start,
        next_segment.end,
        current_segment.start_position,  # Outer boundaries
        next_segment.end_position
    )
```

**Changed from previous gap-based logic** (gap ≤ 1.5s) to **overlap-based logic** (overlapping timestamps).

---

## Data Structures

### TaskSegmentResult

```python
@dataclass
class TaskSegmentResult:
    segmentation_id: str              # Format: "seg_task_001", "seg_task_002", ...
    start_timestamp: int              # Nanoseconds (int64)
    finish_timestamp: int             # Nanoseconds (int64)
    duration_ns: int                  # finish - start in nanoseconds
    start_location: np.ndarray        # [x, y, z] in Unity coordinates (meters)
    finish_location: np.ndarray       # [x, y, z] in Unity coordinates (meters)
    is_location_consolidated: bool    # True if distance between locations < 0.5m

    # Properties
    @property
    def duration_s(self) -> float:
        """Duration in seconds"""
        return self.duration_ns / 1e9

    @property
    def location_distance(self) -> float:
        """Euclidean distance between start and finish locations"""
        return float(np.linalg.norm(self.finish_location - self.start_location))
```

### SegmentationConfig

```python
@dataclass
class SegmentationConfig:
    # Mode selection
    stability_mode: str = "percentage"  # "strict" or "percentage"

    # Optional criteria
    require_spatial_stability: bool = True
    require_gaze_stability: bool = True
    require_hand_tracking: bool = False

    # Spatial criteria
    spatial_stability_radius_m: float = 0.3
    stability_time_window_s: float = 1.5
    # ... (see Configuration section for full list)

    # Percentage thresholds
    head_instability_threshold_percent: float = 20.0
    gaze_instability_threshold_percent: float = 20.0
    hand_untracked_threshold_percent: float = 20.0
    min_samples_for_percentage_mode: int = 5

    @classmethod
    def from_plugin_config(cls, plugin_config: Dict) -> 'SegmentationConfig':
        """Create configuration from plugin config dictionary"""
```

### Input Data Format

#### Organized Session Structure
```
/session_*/
├── metadata.json         # Session metadata with camera list
├── cameras/
│   ├── {camera_name}/   # Each camera directory
│   │   ├── frame_metadata.csv
│   │   ├── gaze_screen_coords.csv (optional)
│   │   └── frames/ or camera_frames.mlcf
├── sensors/
│   ├── gaze_data.csv    # 3D gaze data
│   ├── imu_data.csv     # IMU sensor data (optional)
│   └── hand_gesture_data.csv  # Hand tracking (optional)
```

#### Required CSV Columns

**frame_metadata.csv** (primary camera):
```csv
timestamp,posX,posY,posZ,rotX,rotY,rotZ,rotW
```

**gaze_data.csv**:
```csv
timestamp,dirX,dirY,dirZ,originX,originY,originZ,gazeState
```

**hand_gesture_data.csv** (optional):
```csv
timestamp,handedness,isTracked,palmPosX,palmPosY,palmPosZ,palmRotX,palmRotY,palmRotZ,palmRotW
```

---

## Usage Examples

### Example 1: Basic Segmentation

```python
from src.Segmentation import TaskSegmenter, SegmentationConfig

config = SegmentationConfig()
segmenter = TaskSegmenter(config)
results = segmenter.segment(session_data)

for segment in results:
    print(f"{segment.segmentation_id}: {segment.duration_s:.1f}s at "
          f"({segment.start_location[0]:.2f}, {segment.start_location[2]:.2f})")
```

### Example 2: Strict Mode with Custom Thresholds

```python
config = SegmentationConfig(
    stability_mode="strict",
    spatial_stability_radius_m=0.2,  # Stricter: 20cm instead of 30cm
    gaze_deviation_threshold_deg=15.0,  # Stricter: 15° instead of 20°
    require_hand_tracking=True  # Enable hand tracking requirement
)
```

### Example 3: Lenient Percentage Mode

```python
config = SegmentationConfig(
    stability_mode="percentage",
    head_instability_threshold_percent=30.0,  # Allow 30% unstable
    gaze_instability_threshold_percent=30.0,
    hand_untracked_threshold_percent=40.0,  # Allow 40% no hands
    require_gaze_stability=False  # Disable gaze checks entirely
)
```

### Example 4: From YAML Configuration

```python
import yaml

with open('config/default_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = SegmentationConfig.from_plugin_config(
    config_dict.get('task_segmenter', {})
)
segmenter = TaskSegmenter(config)
```

---

## Export Formats

### CSV Export

**File**: `task_segments.csv`

```csv
segmentation_id,start_timestamp,finish_timestamp,start_time_from_recording_s,finish_time_from_recording_s,duration_ns,duration_s,start_x,start_y,start_z,finish_x,finish_y,finish_z,location_distance_m,is_location_consolidated
seg_task_001,43511047315,53342867998,43.511,53.343,9831820683,9.832,-0.493,0.026,0.896,-0.564,0.019,0.911,0.072,True
seg_task_002,63109793839,75709858168,63.110,75.710,12600064329,12.600,-1.743,-0.069,-4.444,-2.005,-0.067,-4.631,0.322,True
```

### JSON Export

**File**: `task_segments.json`

```json
{
  "metadata": {
    "format_version": "1.0",
    "export_type": "task_segmentation",
    "coordinate_system": "Unity (left-handed, Y-up)",
    "timestamp_unit": "nanoseconds",
    "position_unit": "meters",
    "total_segments": 2
  },
  "segments": [
    {
      "segmentation_id": "seg_task_001",
      "start_timestamp": 43511047315,
      "finish_timestamp": 53342867998,
      "duration_s": 9.832,
      "start_location": {"x": -0.493, "y": 0.026, "z": 0.896},
      "finish_location": {"x": -0.564, "y": 0.019, "z": 0.911},
      "location_distance_m": 0.072,
      "is_location_consolidated": true
    }
  ]
}
```

### Summary Export

**File**: `task_segments_summary.json`

```json
{
  "metadata": {
    "format_version": "1.0",
    "export_type": "task_segmentation_summary"
  },
  "summary": {
    "total_segments": 4,
    "total_duration_s": 32.465,
    "average_duration_s": 8.116,
    "median_duration_s": 9.016,
    "shortest_duration_s": 1.833,
    "longest_duration_s": 12.600,
    "consolidated_segments": 3,
    "consolidation_rate": 75.0,
    "average_location_distance_m": 0.321,
    "temporal_coverage": {
      "start_timestamp": 43511047315,
      "end_timestamp": 96110388432,
      "time_span_s": 52.599
    }
  }
}
```

### Export Function

```python
from src.Segmentation import export_task_segments

exported_files = export_task_segments(
    results=segment_results,
    output_dir="path/to/output",
    formats=['csv', 'json', 'summary'],  # Choose formats
    recording_start_timestamp=session.start_timestamp  # Optional
)

# Returns dict with file paths
print(exported_files['csv'])     # Path to CSV file
print(exported_files['json'])    # Path to JSON file
print(exported_files['summary']) # Path to summary file
```

---

## Testing

### Test Coverage

**Total Tests**: 34 tests passing
- 24 tests for core segmentation (TaskSegmenter)
- 10 tests for percentage mode (TestPercentageMode)

**Test Categories**:
1. Initialization and configuration
2. Session data validation
3. Camera position extraction
4. Spatial stability (strict and percentage modes)
5. Gaze stability (strict and percentage modes)
6. Hand tracking (strict and percentage modes)
7. Segment creation and merging
8. Time buffer application
9. Location consolidation
10. Result generation
11. Export functionality
12. Mode switching and fallback behavior
13. Edge cases and boundary conditions

### Running Tests

```bash
# Run all segmentation tests
uv run pytest tests/test_segmenter.py -v

# Run specific test class
uv run pytest tests/test_segmenter.py::TestPercentageMode -v

# Run with output
uv run pytest tests/test_segmenter.py::TestTaskSegmenter::test_export_task_segments_to_test_dir -v -s
```

### Test Data

**Location**: `/test_dir/`
- Real ML2 session data with camera, gaze, and hand tracking
- Used for integration and export testing
- Results exported to `/test_dir/Segmented_Tasks/`

---

## File Structure

```
src/Segmentation/
├── __init__.py              # Module exports
├── README.md                # This file
├── Todo.md                  # Development history (deprecated, see README)
├── data_types.py            # TaskSegmentResult and SegmentationConfig
├── segmenter.py             # Main TaskSegmenter implementation
└── export.py                # Export functionality

tests/
└── test_segmenter.py        # Comprehensive test suite (34 tests)

config/
└── default_config.yaml      # Default configuration with task_segmenter section

test_dir/
├── metadata.json            # Test session metadata
├── cameras/                 # Test camera data
├── sensors/                 # Test gaze and hand tracking data
└── Segmented_Tasks/         # Export output directory
```

---

## Hand Tracking Integration

### Data Format

Hand tracking data is stored in `sensors/hand_gesture_data.csv`:

```csv
timestamp,handedness,isTracked,palmPosX,palmPosY,palmPosZ,palmRotX,palmRotY,palmRotZ,palmRotW
1705000000000000000,Left,true,0.1,0.2,0.3,0.0,0.0,0.0,1.0
1705000000000000000,Right,true,0.4,0.5,0.6,0.0,0.0,0.0,1.0
```

### Validation Rules

- **Requirement**: At least one hand (Left OR Right) must have `isTracked = true` at each timestamp
- **Percentage Mode**: Allow up to 20% (configurable) of timestamps to have no tracking
- **Strict Mode**: ALL timestamps must have at least one tracked hand
- **Auto-disable**: If hand_gesture_data.csv is missing, hand tracking validation is automatically disabled
- **Optional**: Can be enabled/disabled via `require_hand_tracking` configuration

### Integration with Segmentation

- Hand tracking works like spatial/gaze stability - contributes to window stability evaluation
- Timestamps failing hand tracking don't immediately break segments
- Brief tracking losses handled by merge threshold (1.5s gap)
- All segmentation criteria are optional and independently configurable

---

## Coordinate System

**Unity Left-Handed, Y-Up**:
- X: Right/Left (horizontal)
- Y: Up/Down (vertical) - **IGNORED in spatial stability calculations**
- Z: Forward/Backward (horizontal)
- Units: Meters

**Spatial Stability Distance**: Only X and Z components used to prevent vertical head bobbing from breaking stability.

---

## Notes and Best Practices

### When to Use Percentage Mode
- Default choice for most applications
- Noisy sensor data (camera jitter, tracking dropouts)
- Real-world recording conditions
- When you want more segments detected

### When to Use Strict Mode
- High-quality controlled environment data
- When you need very high confidence segments
- Critical applications requiring zero tolerance
- When fewer, higher-quality segments preferred

### Tuning Parameters

**For More Segments**:
- Use percentage mode
- Increase percentage thresholds (e.g., 30%)
- Decrease spatial_stability_radius_m
- Increase gaze_deviation_threshold_deg
- Disable optional criteria

**For Fewer, Higher-Quality Segments**:
- Use strict mode
- Decrease percentage thresholds (e.g., 10%)
- Increase spatial_stability_radius_m
- Decrease gaze_deviation_threshold_deg
- Enable all criteria

### Performance Considerations

- Sliding window is O(n*w) where n=timestamps, w=window_size
- Percentage calculations add minimal overhead
- For large sessions (>10 minutes), expect ~1-5 seconds processing time
- Export operations are fast (<1 second for typical sessions)

---

## Version History

### Version 2.0 (Current)
- Added percentage-based stability mode
- Implemented gaze state filtering (Blink/Unknown removal)
- Changed to overlap-based segment merging
- Modified spatial distance to use X-Z plane only (ignore Y)
- Added comprehensive test suite for percentage mode
- Updated configuration with new parameters
- Improved documentation and examples

### Version 1.0
- Initial implementation with strict mode only
- Basic spatial, gaze, and hand tracking validation
- Gap-based segment merging
- CSV/JSON export functionality
- 24 core tests passing

---

## References

- Rerun SDK: https://rerun.io/docs
- Unity Coordinate System: Left-handed, Y-up
- Magic Leap 2 Sensor Data Format
- MLGaze Viewer Architecture Documentation
