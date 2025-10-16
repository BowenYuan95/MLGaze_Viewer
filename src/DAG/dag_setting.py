"""DAG setting functions for managing dependencies between segmentation tasks."""

import numpy as np
from typing import List, Dict, Optional

from src.Segmentation.data_types import TaskSegmentResult


def set_dependency(child_segment_id: str, parent_segment_id: str, segments: List[TaskSegmentResult]) -> None:
    """Add a dependency relationship with automatically calculated edge costs.

    Calculates distance (horizontal X-Z plane) and time difference between parent finish
    and child start, then stores in child_segment's dependency fields.

    Args:
        child_segment_id: ID of the child segment that depends on the parent
        parent_segment_id: ID of the parent segment
        segments: List of all TaskSegmentResult objects

    Raises:
        ValueError: If segment IDs not found, dependency already exists, or temporal order is invalid
    """
    # Find child segment by ID
    child_segment = None
    for seg in segments:
        if seg.segmentation_id == child_segment_id:
            child_segment = seg
            break

    if child_segment is None:
        raise ValueError(f"Child segment with ID '{child_segment_id}' not found in segments list")

    # Find parent segment by ID
    parent_segment = None
    for seg in segments:
        if seg.segmentation_id == parent_segment_id:
            parent_segment = seg
            break

    if parent_segment is None:
        raise ValueError(f"Parent segment with ID '{parent_segment_id}' not found in segments list")

    # Check if dependency already exists
    if parent_segment_id in child_segment.dependency_segment_ids:
        raise ValueError(f"Dependency '{parent_segment_id}' already exists in segment '{child_segment_id}'")

    # Validate temporal ordering: parent must finish before child starts
    if parent_segment.finish_timestamp > child_segment.start_timestamp:
        parent_finish_s = parent_segment.finish_timestamp / 1e9
        child_start_s = child_segment.start_timestamp / 1e9
        raise ValueError(
            f"Invalid dependency: parent '{parent_segment_id}' finishes at {parent_finish_s:.3f}s "
            f"which is after child '{child_segment_id}' starts at {child_start_s:.3f}s. "
            f"Parent must finish before child starts."
        )

    # Calculate horizontal distance (X-Z plane only, ignoring Y)
    parent_finish = parent_segment.finish_location
    child_start = child_segment.start_location

    horizontal_diff = np.array([
        child_start[0] - parent_finish[0],  # X difference
        child_start[2] - parent_finish[2]   # Z difference
    ])
    distance_m = float(np.linalg.norm(horizontal_diff))

    # Calculate time difference in seconds
    time_diff_ns = child_segment.start_timestamp - parent_segment.finish_timestamp
    time_s = float(time_diff_ns / 1e9)

    # Add dependency
    child_segment.dependency_segment_ids.append(parent_segment_id)

    # Store edge costs
    child_segment.edge_costs[parent_segment_id] = {
        "distance_m": distance_m,
        "time_s": time_s
    }


def remove_dependency(segment_id: str, parent_id: str, segments: List[TaskSegmentResult]) -> None:
    """Remove a dependency relationship from a segment.

    Args:
        segment_id: ID of the segment to remove the dependency from
        parent_id: The parent segment ID to remove
        segments: List of all TaskSegmentResult objects

    Raises:
        ValueError: If segment_id not found or parent_id doesn't exist in dependencies
    """
    # Find segment by ID
    segment = None
    for seg in segments:
        if seg.segmentation_id == segment_id:
            segment = seg
            break

    if segment is None:
        raise ValueError(f"Segment with ID '{segment_id}' not found in segments list")

    if parent_id not in segment.dependency_segment_ids:
        raise ValueError(
            f"Dependency '{parent_id}' not found in segment '{segment_id}'. "
            f"Existing dependencies: {segment.dependency_segment_ids}"
        )

    # Remove from dependency list
    segment.dependency_segment_ids.remove(parent_id)

    # Remove from edge costs
    if parent_id in segment.edge_costs:
        del segment.edge_costs[parent_id]


def get_dependencies(segment_id: str, segments: List[TaskSegmentResult]) -> List[str]:
    """Get list of parent segment IDs that this segment depends on.

    Args:
        segment_id: ID of the segment to query
        segments: List of all TaskSegmentResult objects

    Returns:
        List of parent segment IDs (empty list if no dependencies)

    Raises:
        ValueError: If segment_id not found
    """
    # Find segment by ID
    segment = None
    for seg in segments:
        if seg.segmentation_id == segment_id:
            segment = seg
            break

    if segment is None:
        raise ValueError(f"Segment with ID '{segment_id}' not found in segments list")

    return segment.dependency_segment_ids.copy()


def get_edge_distance(segment_id: str, parent_id: str, segments: List[TaskSegmentResult]) -> Optional[float]:
    """Get the distance cost for a specific dependency edge.

    Args:
        segment_id: ID of the segment containing the dependency
        parent_id: The parent segment ID
        segments: List of all TaskSegmentResult objects

    Returns:
        Distance in meters, or None if dependency doesn't exist

    Raises:
        ValueError: If segment_id not found
    """
    # Find segment by ID
    segment = None
    for seg in segments:
        if seg.segmentation_id == segment_id:
            segment = seg
            break

    if segment is None:
        raise ValueError(f"Segment with ID '{segment_id}' not found in segments list")

    if parent_id not in segment.edge_costs:
        return None

    return segment.edge_costs[parent_id].get("distance_m")


def get_edge_time(segment_id: str, parent_id: str, segments: List[TaskSegmentResult]) -> Optional[float]:
    """Get the time cost for a specific dependency edge.

    Args:
        segment_id: ID of the segment containing the dependency
        parent_id: The parent segment ID
        segments: List of all TaskSegmentResult objects

    Returns:
        Time in seconds, or None if dependency doesn't exist

    Raises:
        ValueError: If segment_id not found
    """
    # Find segment by ID
    segment = None
    for seg in segments:
        if seg.segmentation_id == segment_id:
            segment = seg
            break

    if segment is None:
        raise ValueError(f"Segment with ID '{segment_id}' not found in segments list")

    if parent_id not in segment.edge_costs:
        return None

    return segment.edge_costs[parent_id].get("time_s")


def get_edge_cost(segment_id: str, parent_id: str, segments: List[TaskSegmentResult]) -> Optional[Dict[str, float]]:
    """Get both distance and time costs for a specific dependency edge.

    Args:
        segment_id: ID of the segment containing the dependency
        parent_id: The parent segment ID
        segments: List of all TaskSegmentResult objects

    Returns:
        Dictionary with 'distance_m' and 'time_s' keys, or None if dependency doesn't exist

    Raises:
        ValueError: If segment_id not found
    """
    # Find segment by ID
    segment = None
    for seg in segments:
        if seg.segmentation_id == segment_id:
            segment = seg
            break

    if segment is None:
        raise ValueError(f"Segment with ID '{segment_id}' not found in segments list")

    if parent_id not in segment.edge_costs:
        return None

    return segment.edge_costs[parent_id].copy()
