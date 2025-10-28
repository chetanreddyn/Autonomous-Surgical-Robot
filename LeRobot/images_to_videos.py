#!/usr/bin/env python3
import cv2
import os
import re
import numpy as np
import argparse
import logging
from glob import glob
from datetime import datetime
from typing import List, Sequence, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("VideoGen")


class VideoGenerator:
    def __init__(self, fps: int = 30, codec: str = "mp4v"):
        """
        Args:
            fps: frames per second for output videos
            codec: fourcc string for cv2.VideoWriter (e.g., 'mp4v', 'XVID')
        """
        self.fps = fps
        self.codec = codec

    @staticmethod
    def _extract_timestamp_from_name(filename: str, camera_name: str) -> datetime:
        """Extracts timestamp from filename like camera_left_YYYY-MM-DD HH:MM:SS.sss.png"""
        base = os.path.basename(filename)
        pattern = rf"{re.escape(camera_name)}_(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}\.\d+)"
        m = re.search(pattern, base)
        if not m:
            raise ValueError(f"Timestamp parse failed for file: {base}")
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")

    def get_image_filenames(self, images_dir: str, camera_name: str) -> List[str]:
        """Return sorted list of image file paths for camera_name."""
        pattern = os.path.join(images_dir, f"{camera_name}_*.npy")
        files = glob(pattern)
        if not files:
            logger.warning("No images found for pattern: %s", pattern)
            return []
        try:
            files.sort(key=lambda f: self._extract_timestamp_from_name(f, camera_name))
        except ValueError as e:
            logger.error("Error parsing timestamps: %s", e)
            # fallback to lexical sort
            files.sort()
        return files

    def create_video_from_images(self, image_files: Sequence[str], output_path: str) -> bool:
        """Create MP4 video from ordered image list. Returns True on success."""
        if not image_files:
            logger.warning("No image files provided for %s", output_path)
            return False

        first = np.load(image_files[0])
        if first is None:
            logger.error("Could not read first image: %s", image_files[0])
            return False
        height, width = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        if not writer.isOpened():
            logger.error("VideoWriter failed to open for %s (codec=%s)", output_path, self.codec)
            return False

        for img_path in image_files:
            img = np.load(img_path)
            if img is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height))
            writer.write(img)

        writer.release()
        # logger.info("Saved video: %s", output_path)
        return True

    def generate_videos(self, demo_dir: str, cameras: Sequence[str] = ("camera_left", "camera_right")) -> None:
        """Generate videos for a single demo directory."""
        images_dir = os.path.join(demo_dir, "images")
        output_videos_dir = os.path.join(demo_dir, "videos")
        os.makedirs(output_videos_dir, exist_ok=True)
        for camera in cameras:
            files = self.get_image_filenames(images_dir, camera)
            if not files:
                logger.info("No files for %s in %s", camera, demo_dir)
                continue
            out_path = os.path.join(output_videos_dir, f"{camera}.mp4")
            self.create_video_from_images(files, out_path)

    def generate_videos_for_all_demos(
        self, exp_dir: str, demo_start: int, demo_end: int, cameras: Sequence[str] = ("camera_left", "camera_right")
    ) -> None:
        """Generate videos for Demo{N} directories within exp_dir inclusive."""
        for i in range(demo_start, demo_end + 1):
            demo_dir = os.path.join(exp_dir, f"Demo{i}")
            if not os.path.isdir(demo_dir):
                logger.warning("Demo directory not found: %s", demo_dir)
                continue
            logger.info("Processing Demo %s", i)
            self.generate_videos(demo_dir, cameras=cameras)


if __name__ == "__main__":
    # args = parse_args()
    exp_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan"
    demo_start = 1
    demo_end = 2
    fps = 30
    codec = "mp4v"
    vg = VideoGenerator(fps, codec)
    vg.generate_videos_for_all_demos(exp_dir, demo_start=demo_start, demo_end=demo_end)