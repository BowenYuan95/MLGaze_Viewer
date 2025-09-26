"""Export functionality for task segmentation results."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np

from .data_types import TaskSegmentResult


class SegmentationExporter:
    """Handles exporting TaskSegmentResult data to various formats."""

    @staticmethod
    def to_csv(segments: List[TaskSegmentResult], output_path: Union[str, Path],
               recording_start_timestamp: int = None) -> None:
        """Export segmentation results to CSV format.

        Args:
            segments: List of TaskSegmentResult objects
            output_path: Path to output CSV file
            recording_start_timestamp: Timestamp of recording start in nanoseconds (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not segments:
            # Create empty CSV with headers
            headers = [
                'segmentation_id', 'start_timestamp', 'finish_timestamp', 'duration_ns',
                'duration_s', 'start_x', 'start_y', 'start_z',
                'finish_x', 'finish_y', 'finish_z',
                'location_distance_m', 'is_location_consolidated'
            ]

            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            return

        # Convert segments to rows
        rows = []
        for segment in segments:
            # Calculate times from recording beginning (in seconds)
            if recording_start_timestamp is not None:
                start_time_from_recording_s = (segment.start_timestamp - recording_start_timestamp) / 1e9
                finish_time_from_recording_s = (segment.finish_timestamp - recording_start_timestamp) / 1e9
            else:
                start_time_from_recording_s = segment.start_timestamp / 1e9
                finish_time_from_recording_s = segment.finish_timestamp / 1e9

            row = [
                segment.segmentation_id,
                segment.start_timestamp,
                segment.finish_timestamp,
                start_time_from_recording_s,
                finish_time_from_recording_s,
                segment.duration_ns,
                segment.duration_s,
                float(segment.start_location[0]),
                float(segment.start_location[1]),
                float(segment.start_location[2]),
                float(segment.finish_location[0]),
                float(segment.finish_location[1]),
                float(segment.finish_location[2]),
                segment.location_distance,
                segment.is_location_consolidated
            ]
            rows.append(row)

        # Write to CSV
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow([
                'segmentation_id', 'start_timestamp', 'finish_timestamp',
                'start_time_from_recording_s', 'finish_time_from_recording_s',
                'duration_ns', 'duration_s', 'start_x', 'start_y', 'start_z',
                'finish_x', 'finish_y', 'finish_z',
                'location_distance_m', 'is_location_consolidated'
            ])

            # Write data rows
            writer.writerows(rows)

    @staticmethod
    def to_json(segments: List[TaskSegmentResult], output_path: Union[str, Path],
                recording_start_timestamp: int = None) -> None:
        """Export segmentation results to JSON format.

        Args:
            segments: List of TaskSegmentResult objects
            output_path: Path to output JSON file
            recording_start_timestamp: Timestamp of recording start in nanoseconds (optional)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert segments to serializable format
        segments_data = []
        for segment in segments:
            # Calculate times from recording beginning (in seconds)
            if recording_start_timestamp is not None:
                start_time_from_recording_s = (segment.start_timestamp - recording_start_timestamp) / 1e9
                finish_time_from_recording_s = (segment.finish_timestamp - recording_start_timestamp) / 1e9
            else:
                start_time_from_recording_s = segment.start_timestamp / 1e9
                finish_time_from_recording_s = segment.finish_timestamp / 1e9

            segment_dict = {
                'segmentation_id': segment.segmentation_id,
                'start_timestamp': int(segment.start_timestamp),
                'finish_timestamp': int(segment.finish_timestamp),
                'start_time_from_recording_s': float(start_time_from_recording_s),
                'finish_time_from_recording_s': float(finish_time_from_recording_s),
                'duration_ns': int(segment.duration_ns),
                'duration_s': float(segment.duration_s),
                'start_location': {
                    'x': float(segment.start_location[0]),
                    'y': float(segment.start_location[1]),
                    'z': float(segment.start_location[2])
                },
                'finish_location': {
                    'x': float(segment.finish_location[0]),
                    'y': float(segment.finish_location[1]),
                    'z': float(segment.finish_location[2])
                },
                'location_distance_m': float(segment.location_distance),
                'is_location_consolidated': bool(segment.is_location_consolidated)
            }
            segments_data.append(segment_dict)

        # Create export metadata
        export_data = {
            'metadata': {
                'format_version': '1.0',
                'export_type': 'task_segmentation',
                'coordinate_system': 'Unity (left-handed, Y-up)',
                'timestamp_unit': 'nanoseconds',
                'position_unit': 'meters',
                'total_segments': len(segments)
            },
            'segments': segments_data
        }

        # Write to JSON with proper formatting
        with open(output_path, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)

    @staticmethod
    def to_dataframe(segments: List[TaskSegmentResult]) -> pd.DataFrame:
        """Convert segmentation results to pandas DataFrame.

        Args:
            segments: List of TaskSegmentResult objects

        Returns:
            DataFrame with segmentation data
        """
        if not segments:
            # Return empty DataFrame with proper columns
            return pd.DataFrame(columns=[
                'segmentation_id', 'start_timestamp', 'finish_timestamp', 'duration_ns',
                'duration_s', 'start_x', 'start_y', 'start_z',
                'finish_x', 'finish_y', 'finish_z',
                'location_distance_m', 'is_location_consolidated'
            ])

        # Convert to list of dictionaries
        data = []
        for segment in segments:
            row = {
                'segmentation_id': segment.segmentation_id,
                'start_timestamp': segment.start_timestamp,
                'finish_timestamp': segment.finish_timestamp,
                'duration_ns': segment.duration_ns,
                'duration_s': segment.duration_s,
                'start_x': float(segment.start_location[0]),
                'start_y': float(segment.start_location[1]),
                'start_z': float(segment.start_location[2]),
                'finish_x': float(segment.finish_location[0]),
                'finish_y': float(segment.finish_location[1]),
                'finish_z': float(segment.finish_location[2]),
                'location_distance_m': segment.location_distance,
                'is_location_consolidated': segment.is_location_consolidated
            }
            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def export_summary(segments: List[TaskSegmentResult], output_path: Union[str, Path]) -> None:
        """Export a summary of segmentation results.

        Args:
            segments: List of TaskSegmentResult objects
            output_path: Path to output summary file (JSON format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not segments:
            summary = {
                'total_segments': 0,
                'total_duration_s': 0.0,
                'average_duration_s': 0.0,
                'median_duration_s': 0.0,
                'shortest_duration_s': 0.0,
                'longest_duration_s': 0.0,
                'consolidated_segments': 0,
                'consolidation_rate': 0.0,
                'average_location_distance_m': 0.0,
                'temporal_coverage': {
                    'start_timestamp': None,
                    'end_timestamp': None,
                    'time_span_s': 0.0
                }
            }
        else:
            durations = [seg.duration_s for seg in segments]
            distances = [seg.location_distance for seg in segments]
            consolidated_count = sum(1 for seg in segments if seg.is_location_consolidated)

            summary = {
                'total_segments': len(segments),
                'total_duration_s': float(sum(durations)),
                'average_duration_s': float(np.mean(durations)),
                'median_duration_s': float(np.median(durations)),
                'shortest_duration_s': float(min(durations)),
                'longest_duration_s': float(max(durations)),
                'consolidated_segments': consolidated_count,
                'consolidation_rate': float(consolidated_count / len(segments) * 100),
                'average_location_distance_m': float(np.mean(distances)),
                'temporal_coverage': {
                    'start_timestamp': int(min(seg.start_timestamp for seg in segments)),
                    'end_timestamp': int(max(seg.finish_timestamp for seg in segments)),
                    'time_span_s': float((max(seg.finish_timestamp for seg in segments) -
                                        min(seg.start_timestamp for seg in segments)) / 1e9)
                }
            }

        # Add metadata
        export_data = {
            'metadata': {
                'format_version': '1.0',
                'export_type': 'task_segmentation_summary',
                'timestamp_unit': 'nanoseconds',
                'position_unit': 'meters'
            },
            'summary': summary
        }

        with open(output_path, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2, ensure_ascii=False)


def export_task_segments(segments: List[TaskSegmentResult],
                        output_dir: Union[str, Path],
                        formats: List[str] = None,
                        include_summary: bool = True,
                        recording_start_timestamp: int = None) -> Dict[str, Path]:
    """Convenience function to export segmentation results in multiple formats.

    Args:
        segments: List of TaskSegmentResult objects
        output_dir: Directory to save export files
        formats: List of formats to export ('csv', 'json', 'summary'). Default: all
        include_summary: Whether to include summary export
        recording_start_timestamp: Timestamp of recording start in nanoseconds (optional)

    Returns:
        Dictionary mapping format names to output file paths
    """
    if formats is None:
        formats = ['csv', 'json']

    if include_summary and 'summary' not in formats:
        formats.append('summary')

    output_dir = Path(output_dir) / "Segmented_Tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {}
    exporter = SegmentationExporter()

    for format_name in formats:
        if format_name == 'csv':
            output_path = output_dir / 'task_segments.csv'
            exporter.to_csv(segments, output_path, recording_start_timestamp)
            exported_files['csv'] = output_path

        elif format_name == 'json':
            output_path = output_dir / 'task_segments.json'
            exporter.to_json(segments, output_path, recording_start_timestamp)
            exported_files['json'] = output_path

        elif format_name == 'summary':
            output_path = output_dir / 'task_segments_summary.json'
            exporter.export_summary(segments, output_path)
            exported_files['summary'] = output_path

    return exported_files


# Add convenience methods to TaskSegmentResult class
def _add_export_methods():
    """Add export methods to TaskSegmentResult class."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskSegmentResult to dictionary."""
        return {
            'segmentation_id': self.segmentation_id,
            'start_timestamp': int(self.start_timestamp),
            'finish_timestamp': int(self.finish_timestamp),
            'duration_ns': int(self.duration_ns),
            'duration_s': float(self.duration_s),
            'start_location': {
                'x': float(self.start_location[0]),
                'y': float(self.start_location[1]),
                'z': float(self.start_location[2])
            },
            'finish_location': {
                'x': float(self.finish_location[0]),
                'y': float(self.finish_location[1]),
                'z': float(self.finish_location[2])
            },
            'location_distance_m': float(self.location_distance),
            'is_location_consolidated': bool(self.is_location_consolidated)
        }

    # Add methods to TaskSegmentResult class
    TaskSegmentResult.to_dict = to_dict

# Apply the methods
_add_export_methods()