# Task-Based Session Segmentation Implementation

## Project Overview

This module implements task-based session segmentation for MLGaze Viewer, identifying spatially stable and visually focused periods within Magic Leap 2 recording sessions.

### Segmentation Criteria
A task segment is defined when the recorder:
- Remains spatially stable within a **0.3m radius** using **1.5s sliding window**
- Focus point remains within **1.5m** from head position
- Maintains gaze deviation **≤ 20°** between consecutive samples
- Uses smoothed gaze rays (low-confidence samples discarded)
- Applies **1s exponential moving average**, sampled every **500ms**

### Signal Processing
- Head trajectories: Low-pass filtered (cutoff ≈ 1.5 Hz) using exponential smoother (α ≈ 0.48 at 10Hz)
- Outlier rejection: Displacements > 1.5m/100ms or speeds > 3 m/s are discarded
- Missing sensor values: Interpolated with zero-order hold up to 100ms
- Long gaps: > 300ms gaps are excluded
- Segment merging: Consecutive candidates < 1.5s apart are merged

## Implementation Tasks

### ✅ Completed Tasks
- [x] Project requirements analysis and clarification
- [x] Todo.md creation with comprehensive plan and hand tracking integration
- [x] **Task 1**: Create `__init__.py` with proper exports following codebase patterns
- [x] **Task 2**: Create `data_types.py` with TaskSegmentResult and SegmentationConfig
- [x] **Task 3**: Add `task_segmenter` section to `config/default_config.yaml`
- [x] **Task 4**: Update `VisualizationConfig` class with segmentation parameters
- [x] **Task 5**: Update configuration with optional criteria and simplified hand tracking

### ✅ Foundation Implementation
- [x] **Task 6**: Add `hand_tracking` field to SessionData class
- [x] **Task 7**: Create `signal_processing.py` with filtering and interpolation utilities
- [x] **Task 8**: Create `segmenter.py` with main TaskSegmenter class

### ✅ Core Algorithm Features
- [x] **Task 9**: Add `get_task_segments()` method to SessionData class
- [x] **Task 10**: Implement spatial stability detection using primary camera position
- [x] **Task 11**: Implement gaze analysis with state filtering and deviation calculation
- [x] **Task 12**: Implement simplified hand tracking validation (no timeout)

### ✅ Advanced Processing
- [x] **Task 13**: Implement location consolidation logic (< 0.5m threshold)
- [x] **Task 14**: Implement segment merging and short segment handling
- [x] **Task 15**: Create CSV/JSON export functionality for TaskSegmentResult

### ✅ Testing and Validation
- [x] **Task 16**: Unit tests for signal processing utilities
- [x] **Task 17**: Unit tests for segmentation algorithm components - All 58 tests passing
- [x] **Task 18**: Integration tests with real SessionData - 10 comprehensive integration tests
- [x] **Task 19**: Performance and edge case testing - 9 performance and edge case tests

**Total Test Coverage**: 77 tests passing - Complete system validation achieved!

## Hand Tracking Integration

### Hand Tracking Data Format
Hand tracking data is stored in `sensors/hand_gesture_data.csv` with the following structure:

```csv
timestamp,handedness,isTracked,palmPosX,palmPosY,palmPosZ,palmRotX,palmRotY,palmRotZ,palmRotW
1705000000000000000,Left,true,0.1,0.2,0.3,0.0,0.0,0.0,1.0
1705000000000000000,Right,true,0.4,0.5,0.6,0.0,0.0,0.0,1.0
1705000033333333333,Left,false,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1705000033333333333,Right,true,0.4,0.5,0.6,0.0,0.0,0.0,1.0
```

### Hand Tracking Validation Rules
- **Requirement**: At least one hand (Left OR Right) must have `isTracked = true` during task segments
- **Stability marking**: Timestamps without hand tracking data OR with both hands having `isTracked = false` are marked as unstable
- **No immediate termination**: Brief tracking losses (< 1.5s) are handled by merge threshold, not immediate segment termination
- **Auto-disable**: If `hand_gesture_data.csv` is missing, disable hand tracking validation automatically
- **Optional**: Hand tracking validation can be enabled/disabled via configuration

### Integration with Segmentation
- Hand tracking works like spatial stability - creates gaps in stability, not hard boundaries
- Timestamps failing hand tracking validation are marked as "not in task" but don't break segments
- Merge threshold (1.5s gap) handles brief tracking losses from any cause (hand occlusion, spatial movement, gaze deviation)
- All segmentation criteria are optional and configurable

## Data Structures

### TaskSegmentResult
```python
@dataclass
class TaskSegmentResult:
    segmentation_id: str              # Format: "seg_task_001"
    start_timestamp: int              # Nanoseconds
    finish_timestamp: int             # Nanoseconds  
    duration_ns: int                  # finish - start
    start_location: np.ndarray        # [x, y, z] Unity coordinates
    finish_location: np.ndarray       # [x, y, z] Unity coordinates
    is_location_consolidated: bool    # True if distance < 0.5m
```

## Configuration Parameters

### Global Settings (task_segmenter)
```yaml
task_segmenter:
  enabled: true
  
  # Optional criteria (all can be enabled/disabled)
  require_spatial_stability: true
  require_gaze_stability: true
  require_hand_tracking: false              # Default disabled (optional enhancement)
  
  # Spatial stability criteria
  spatial_stability_radius_m: 0.3
  spatial_stability_min_duration_s: 1.0
  stability_time_window_s: 1.5
  stability_buffer_start_s: 0.5
  stability_buffer_end_s: 0.5

  # Gaze analysis parameters
  gaze_deviation_threshold_deg: 20.0
  focus_distance_threshold_m: 1.5
  gaze_smoothing_window_s: 1.0
  gaze_sampling_interval_s: 0.5
  filtered_gaze_states: ["Blink", "Unknown"]

  # Signal processing
  head_filter_alpha: 0.48
  outlier_displacement_threshold_m: 1.5
  outlier_speed_threshold_ms: 3.0

  # Segment processing
  merge_consecutive_gap_s: 1.5
  location_consolidation_threshold_m: 0.5
  min_segment_duration_s: 1.0
```

## Algorithm Flow

1. **Load SessionData** with primary camera metadata and optional hand tracking data
2. **Extract recorder positions** from camera pose data
2.5. **Apply 1.5s sliding window** for each timestamp in the dataset
3. **Apply signal processing**: Exponential smoothing + outlier rejection
4. **Filter gaze data**: Remove low-confidence states, apply EMA smoothing
5. **Validate hand tracking** (if enabled): Mark timestamps as unstable when no data exists OR both hands have isTracked=false
6. **Detect spatial stability** (if enabled): 0.3m radius within 1.5s sliding window
7. **Detect gaze stability** (if enabled): ≤20° deviation between consecutive samples
7.5. **Check focus distance** (if enabled): Ensure gaze focus point within 1.5m of head position
8. **Identify candidate segments** meeting enabled criteria (spatial + gaze + focus + hand tracking)
8.5. **Apply time buffers**: -0.5s at segment start, +0.5s at segment end (configurable)
9. **Merge consecutive segments** separated by < 1.5s
10. **Handle short segments** with previous/next position comparison
11. **Consolidate locations** if start/finish distance < 0.5m
12. **Generate results** with proper ID formatting and metadata

## Gaze Deviation Calculation

Gaze deviation is calculated as the angle between consecutive gaze direction vectors:

```python
def calculate_gaze_deviation_degrees(gaze_direction_1, gaze_direction_2):
    """Calculate angle between two gaze direction vectors in degrees."""
    # Ensure vectors are normalized
    dir1_normalized = gaze_direction_1 / np.linalg.norm(gaze_direction_1)
    dir2_normalized = gaze_direction_2 / np.linalg.norm(gaze_direction_2)

    # Calculate dot product and clamp to handle numerical errors
    dot_product = np.clip(np.dot(dir1_normalized, dir2_normalized), -1.0, 1.0)

    # Calculate angle in radians then convert to degrees
    angle_radians = np.arccos(dot_product)
    return np.degrees(angle_radians)
```

- Extract gaze direction vectors from consecutive samples within sliding window
- Apply calculation after EMA smoothing with 1.0s window
- Compare result against 20° threshold for stability validation

## Sliding Window Implementation

The sliding window approach ensures robust stability detection by evaluating continuous periods rather than single timestamps:

### Window Processing
1. **For each timestamp** in the dataset:
   - Define a 1.5s time window starting from current timestamp
   - Collect all frames within this window (current_time to current_time + 1.5s)

2. **Stability Evaluation**:
   - **Spatial**: Check if all positions within window stay within 0.3m radius from start position
   - **Gaze**: Verify all consecutive gaze direction pairs have deviation ≤ 20°
   - **Focus**: Ensure all focus points stay within 1.5m from head position
   - **Hand Tracking**: Confirm at least one hand tracked throughout window (if enabled)

3. **Window Sliding**:
   - Move window forward by one frame at a time
   - Each timestamp gets evaluated as the start of its own window
   - Overlapping windows provide smooth stability detection

4. **Stability Marking**:
   - If entire window meets all enabled criteria → mark start timestamp as stable
   - If any criterion fails within window → mark start timestamp as unstable
   - Continue until all timestamps processed

### Time Buffers
- Apply configurable buffers after initial segment detection:
  - **Start buffer**: Extend segment start by -0.5s (or configured value)
  - **End buffer**: Extend segment end by +0.5s (or configured value)
- Buffers capture transition periods and provide smoother segment boundaries

This approach mirrors the Unity implementation while accommodating multiple stability criteria working in parallel.

## File Structure
```
src/Segmentation/
├── __init__.py              # Module exports
├── Todo.md                  # This file
├── data_types.py           # TaskSegmentResult and related classes
├── signal_processing.py    # Filtering and interpolation utilities
└── segmenter.py            # Main TaskSegmenter implementation
```

## Integration Points

- **Input**: `SessionData` (same format as existing system)
- **Configuration**: Via `VisualizationConfig.plugin_configs["task_segmenter"]`
- **Output**: `List[TaskSegmentResult]` via `session.get_task_segments()`
- **Coordinates**: Unity left-handed Y-up (same as codebase)
- **Timestamps**: Nanosecond precision (compatible with existing system)

## Notes

- Conservative approach captures "operating" phases vs movement/scanning
- Uses Unity-aligned parameters: 0.3m spatial radius, 1.5s sliding window, 1.0s merge gap
- Sliding window implementation ensures robust stability detection across multiple criteria
- Hand tracking integration works like other stability criteria - creates gaps, doesn't break segments
- Brief instabilities (< 1.5s) handled gracefully through merge threshold
- Time buffers (-0.5s/+0.5s) capture transition periods for smoother boundaries
- All stability criteria (spatial, gaze, focus, hand tracking) are optional and configurable
- Filters out incidental pauses and movement between locations
- Configurable parameters allow runtime adjustments through Rerun
- Maintains compatibility with existing timestamp synchronization modes
- Follows established codebase patterns and architecture