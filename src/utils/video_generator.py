"""Video generation utility for converting session frames to MP4 videos."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import cv2
from src.utils.logger import logger


class VideoGenerator:
    """Generate MP4 videos from extracted camera frames with optional gaze overlay."""

    def __init__(self, codec: str = 'mp4v', quality: int = 95):
        """Initialize the video generator.

        Args:
            codec: Video codec ('mp4v', 'H264', 'XVID', 'avc1')
            quality: JPEG quality for re-encoding if needed (1-100)
        """
        self.codec = codec
        self.quality = quality
        self.log = logger.get_logger('VideoGenerator')

    def generate_all_videos(
        self,
        session_dir: Path,
        output_dir: Optional[Path] = None,
        fps_override: Optional[float] = None,
        overlay_gaze: bool = False,
        camera_filter: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Generate MP4 videos for all cameras in a session.

        Args:
            session_dir: Path to session directory
            output_dir: Output directory (default: session_dir/videos/)
            fps_override: Override FPS (default: auto-detect from metadata)
            overlay_gaze: Whether to overlay gaze visualization (default: False)
            camera_filter: List of camera names to process (default: all)

        Returns:
            Dictionary mapping camera names to generated video paths
        """
        session_path = Path(session_dir)
        cameras_dir = session_path / "cameras"

        if not cameras_dir.exists():
            self.log.error(f"Cameras directory not found: {cameras_dir}")
            print(f"ERROR: Cameras directory not found: {cameras_dir}")
            return {}

        # Create output directory
        if output_dir is None:
            output_dir = session_path / "videos"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(f"Generating videos for session: {session_path.name}")
        print(f"\n{'='*60}")
        print(f"Video Generation Started")
        print(f"{'='*60}")
        print(f"Session: {session_path.name}")
        print(f"Output: {output_dir}")
        print(f"Gaze overlay: {'Enabled' if overlay_gaze else 'Disabled'}")

        # Find all camera directories
        camera_dirs = [d for d in cameras_dir.iterdir() if d.is_dir()]

        if camera_filter:
            camera_dirs = [d for d in camera_dirs if d.name in camera_filter]

        self.log.info(f"Found {len(camera_dirs)} cameras to process")
        print(f"Cameras found: {len(camera_dirs)}")

        # Generate videos
        generated_videos = {}

        for i, camera_dir in enumerate(camera_dirs, 1):
            camera_name = camera_dir.name
            self.log.info(f"\n--- Processing camera {i}/{len(camera_dirs)}: {camera_name} ---")
            print(f"\n[{i}/{len(camera_dirs)}] Processing camera: {camera_name}")

            # Determine output filename
            if overlay_gaze:
                output_path = output_dir / f"{camera_name}_with_gaze.mp4"
            else:
                output_path = output_dir / f"{camera_name}.mp4"

            success = self.generate_camera_video(
                camera_dir=camera_dir,
                output_path=output_path,
                fps=fps_override,
                overlay_gaze=overlay_gaze
            )

            if success:
                generated_videos[camera_name] = output_path
                self.log.success(f"Generated: {output_path}")
                print(f"✓ Success: {output_path.name}")
            else:
                self.log.error(f"Failed to generate video for {camera_name}")
                print(f"✗ Failed: {camera_name}")

        self.log.success(f"\nVideo generation complete: {len(generated_videos)}/{len(camera_dirs)} videos created")
        print(f"\n{'='*60}")
        print(f"Generation Complete: {len(generated_videos)}/{len(camera_dirs)} videos")
        print(f"{'='*60}\n")

        return generated_videos

    def generate_camera_video(
        self,
        camera_dir: Path,
        output_path: Path,
        fps: Optional[float] = None,
        overlay_gaze: bool = False
    ) -> bool:
        """Generate MP4 video for a single camera.

        Args:
            camera_dir: Path to camera directory containing frames/ and frame_metadata.csv
            output_path: Output video path (.mp4)
            fps: Override FPS (default: auto-detect from metadata)
            overlay_gaze: Whether to overlay gaze visualization

        Returns:
            True if successful, False otherwise
        """
        camera_dir = Path(camera_dir)
        output_path = Path(output_path)

        # Validate inputs
        frames_dir = camera_dir / "frames"
        metadata_path = camera_dir / "frame_metadata.csv"

        if not frames_dir.exists():
            self.log.error(f"Frames directory not found: {frames_dir}")
            print(f"  ERROR: Frames directory not found")
            return False

        if not metadata_path.exists():
            self.log.error(f"Metadata not found: {metadata_path}")
            print(f"  ERROR: Metadata not found")
            return False

        try:
            # Load and sort frame metadata
            metadata = self._load_frame_order(metadata_path)

            if len(metadata) == 0:
                self.log.error("No frames found in metadata")
                print(f"  ERROR: No frames in metadata")
                return False

            # Calculate or use override FPS
            if fps is None:
                fps = self._calculate_fps(metadata)
                self.log.info(f"Auto-detected FPS: {fps:.2f}")
                print(f"  FPS: {fps:.2f} (auto-detected)")
            else:
                self.log.info(f"Using override FPS: {fps:.2f}")
                print(f"  FPS: {fps:.2f} (override)")

            # Load gaze screen coordinates if overlaying
            gaze_coords = None
            if overlay_gaze:
                gaze_coords = self._load_gaze_coords(camera_dir)
                if gaze_coords is None:
                    self.log.warning("Gaze screen coordinates not found, skipping overlay")
                    print(f"  WARNING: No gaze data, skipping overlay")
                    overlay_gaze = False

            # Get first frame to determine resolution
            first_frame_id = metadata.iloc[0]['frameId']
            first_frame_path = self._find_frame_path(frames_dir, first_frame_id)

            if first_frame_path is None:
                self.log.error(f"First frame not found: {first_frame_id}")
                print(f"  ERROR: First frame not found")
                return False

            first_img = cv2.imread(str(first_frame_path))
            if first_img is None:
                self.log.error(f"Failed to read first frame: {first_frame_path}")
                print(f"  ERROR: Cannot read first frame")
                return False

            height, width = first_img.shape[:2]
            self.log.info(f"Resolution: {width}×{height}")
            self.log.info(f"Total frames: {len(metadata)}")
            print(f"  Resolution: {width}×{height}")
            print(f"  Frames: {len(metadata)}")

            # Create video writer
            writer = self._create_video_writer(output_path, width, height, fps)

            if writer is None:
                self.log.error("Failed to create video writer")
                print(f"  ERROR: Failed to create video writer")
                return False

            # Write frames
            self._write_frames(
                writer=writer,
                frames_dir=frames_dir,
                metadata=metadata,
                overlay_gaze=overlay_gaze,
                gaze_coords=gaze_coords
            )

            # Release writer
            writer.release()

            # Verify output
            if not output_path.exists():
                self.log.error("Output video was not created")
                print(f"  ERROR: Output file not created")
                return False

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.log.info(f"Output size: {file_size_mb:.2f} MB")
            print(f"  Size: {file_size_mb:.2f} MB")

            return True

        except Exception as e:
            self.log.error(f"Failed to generate video: {e}")
            print(f"  ERROR: {e}")
            return False

    def _load_frame_order(self, metadata_path: Path) -> pd.DataFrame:
        """Load and sort frame metadata by timestamp.

        Args:
            metadata_path: Path to frame_metadata.csv

        Returns:
            Sorted dataframe with frameId and timestamp columns
        """
        metadata = pd.read_csv(metadata_path)

        # Remove BOM if present
        metadata.columns = metadata.columns.str.replace('﻿', '')

        # Ensure required columns exist
        if 'frameId' not in metadata.columns or 'timestamp' not in metadata.columns:
            self.log.error(f"Missing required columns in metadata: {list(metadata.columns)}")
            return pd.DataFrame()

        # Sort by timestamp
        metadata = metadata.sort_values('timestamp').reset_index(drop=True)

        return metadata

    def _calculate_fps(self, metadata: pd.DataFrame) -> float:
        """Calculate FPS from frame timestamps and round to standard values.

        Args:
            metadata: Frame metadata with timestamp column

        Returns:
            Calculated FPS rounded to nearest standard (15, 30, 60)
        """
        if len(metadata) < 2:
            return 30.0

        timestamps = metadata['timestamp'].values
        time_diffs = np.diff(timestamps) / 1e9  # nanoseconds to seconds

        # Use median to avoid outliers
        avg_interval = np.median(time_diffs)

        if avg_interval > 0:
            calculated_fps = 1.0 / avg_interval

            # Round to nearest standard FPS
            standard_fps = [15.0, 30.0, 60.0]
            nearest_fps = min(standard_fps, key=lambda x: abs(x - calculated_fps))

            return nearest_fps
        else:
            return 30.0

    def _load_gaze_coords(self, camera_dir: Path) -> Optional[pd.DataFrame]:
        """Load gaze screen coordinates for overlay.

        Args:
            camera_dir: Path to camera directory

        Returns:
            Gaze coordinates dataframe or None if not found
        """
        gaze_path = camera_dir / "gaze_screen_coords.csv"

        if not gaze_path.exists():
            return None

        try:
            gaze_df = pd.read_csv(gaze_path)
            gaze_df.columns = gaze_df.columns.str.replace('﻿', '')
            return gaze_df
        except Exception as e:
            self.log.warning(f"Failed to load gaze coordinates: {e}")
            return None

    def _find_frame_path(self, frames_dir: Path, frame_id: str) -> Optional[Path]:
        """Find the actual frame file path.

        Args:
            frames_dir: Directory containing frame images
            frame_id: Frame identifier from metadata

        Returns:
            Path to frame file or None if not found
        """
        # Try direct match
        frame_path = frames_dir / f"{frame_id}.jpg"
        if frame_path.exists():
            return frame_path

        # Try with different extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            frame_path = frames_dir / f"{frame_id}{ext}"
            if frame_path.exists():
                return frame_path

        return None

    def _create_video_writer(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float
    ) -> Optional[cv2.VideoWriter]:
        """Create OpenCV VideoWriter.

        Args:
            output_path: Output video path
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            VideoWriter object or None if creation failed
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():
                self.log.error(f"Failed to open video writer with codec '{self.codec}'")
                return None

            return writer

        except Exception as e:
            self.log.error(f"Failed to create video writer: {e}")
            return None

    def _write_frames(
        self,
        writer: cv2.VideoWriter,
        frames_dir: Path,
        metadata: pd.DataFrame,
        overlay_gaze: bool = False,
        gaze_coords: Optional[pd.DataFrame] = None
    ):
        """Write frames to video in temporal order.

        Args:
            writer: OpenCV VideoWriter
            frames_dir: Directory containing frame images
            metadata: Frame metadata sorted by timestamp
            overlay_gaze: Whether to overlay gaze visualization
            gaze_coords: Gaze screen coordinates dataframe
        """
        total_frames = len(metadata)
        frames_written = 0
        frames_duplicated = 0
        last_valid_frame = None

        for idx, row in metadata.iterrows():
            frame_id = row['frameId']
            timestamp = row['timestamp']

            # Find and load frame
            frame_path = self._find_frame_path(frames_dir, frame_id)

            if frame_path is None:
                # Frame missing - duplicate previous frame to maintain timing
                if last_valid_frame is not None:
                    self.log.warning(f"Frame not found: {frame_id}, duplicating previous frame")
                    img = last_valid_frame.copy()
                    frames_duplicated += 1
                else:
                    self.log.error(f"First frame missing, cannot continue: {frame_id}")
                    continue
            else:
                img = cv2.imread(str(frame_path))

                if img is None:
                    # Failed to read - duplicate previous frame
                    if last_valid_frame is not None:
                        self.log.warning(f"Failed to read frame: {frame_path}, duplicating previous")
                        img = last_valid_frame.copy()
                        frames_duplicated += 1
                    else:
                        self.log.error(f"First frame unreadable, cannot continue: {frame_path}")
                        continue
                else:
                    # Successfully loaded - save as last valid frame
                    last_valid_frame = img.copy()

            # Overlay gaze if requested
            if overlay_gaze and gaze_coords is not None:
                img = self._overlay_gaze_on_frame(img, timestamp, gaze_coords)

            # Write frame
            writer.write(img)
            frames_written += 1

            # Progress logging (both logger and console)
            if frames_written % 100 == 0:
                progress = (frames_written / total_frames) * 100
                msg = f"Progress: {frames_written}/{total_frames} frames ({progress:.1f}%)"
                self.log.info(msg)
                print(f"  {msg}")

        # Final summary
        summary = f"Frames written: {frames_written}, duplicated: {frames_duplicated}"
        self.log.info(summary)
        print(f"  {summary}")

    def _overlay_gaze_on_frame(
        self,
        img: np.ndarray,
        timestamp: int,
        gaze_coords: pd.DataFrame
    ) -> np.ndarray:
        """Draw gaze point on frame.

        Args:
            img: Frame image
            timestamp: Frame timestamp (nanoseconds)
            gaze_coords: Gaze screen coordinates dataframe

        Returns:
            Image with gaze overlay
        """
        # Find closest gaze sample by timestamp
        time_diffs = np.abs(gaze_coords['timestamp'].values - timestamp)
        closest_idx = np.argmin(time_diffs)

        # Only overlay if within 50ms
        if time_diffs[closest_idx] < 50_000_000:  # 50ms in nanoseconds
            row = gaze_coords.iloc[closest_idx]

            if 'screenX' in row and 'screenY' in row:
                x, y = int(row['screenX']), int(row['screenY'])

                # Validate coordinates are within frame
                h, w = img.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    # Draw gaze point (green circle)
                    cv2.circle(img, (x, y), 10, (0, 255, 0), 2)

                    # Draw crosshair
                    cv2.line(img, (x - 20, y), (x + 20, y), (0, 255, 0), 1)
                    cv2.line(img, (x, y - 20), (x, y + 20), (0, 255, 0), 1)

        return img


def main():
    """CLI interface for video generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate MP4 videos from MLGaze session frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'session_dir',
        type=str,
        help='Path to session directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: session_dir/videos/)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Override FPS (default: auto-detect from metadata)'
    )

    parser.add_argument(
        '--overlay-gaze',
        action='store_true',
        help='Generate videos with gaze overlay (default: disabled)'
    )

    parser.add_argument(
        '--cameras',
        nargs='+',
        default=None,
        help='Specific cameras to process (default: all)'
    )

    parser.add_argument(
        '--codec',
        type=str,
        default='mp4v',
        choices=['mp4v', 'H264', 'XVID', 'avc1'],
        help='Video codec'
    )

    args = parser.parse_args()

    # Create generator
    generator = VideoGenerator(codec=args.codec)

    # Generate videos
    videos = generator.generate_all_videos(
        session_dir=Path(args.session_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        fps_override=args.fps,
        overlay_gaze=args.overlay_gaze,
        camera_filter=args.cameras
    )

    # Print final summary
    if videos:
        print(f"Generated video(s):\n")
        for camera_name, video_path in videos.items():
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"  {camera_name}:")
            print(f"    Path: {video_path}")
            print(f"    Size: {file_size_mb:.2f} MB\n")
    else:
        print("No videos were generated.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
