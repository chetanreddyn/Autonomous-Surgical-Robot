#!/usr/bin/env python3
import os
import json
import logging
from glob import glob
from typing import List, Optional, Dict, Sequence, Tuple

import pandas as pd
import pdb

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("MetaFilesGenerator")


class MetaFilesGenerator:
    """
    Generate LeRobot meta files (info.json and episodes.jsonl) by scanning
    parquet files under a dataset directory.

    Args:
        dataset_dir: root directory of the dataset (contains `data/` and `meta/`)
        fps: frames per second to write into info.json
        robot_type: string to write into info.json
        codebase_version: codebase version string for info.json
        task_name: task name to put for each episode (single-task datasets)
        splits: dict to write into info.json['splits']
        image_shape: tuple/list [height, width, channels] used for video/image feature entries
    """

    def __init__(
        self,
        dataset_dir: str,
        fps: int = 30,
        robot_type: str = "daVinci Surgical Robot",
        codebase_version: str = "v2.1",
        task_name: str = "Task",
        splits: Optional[Dict[str, str]] = None,
        image_shape: Optional[Sequence[int]] = (540, 960, 3),
        codec: str = "h264",
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        os.makedirs(self.meta_dir, exist_ok=True)

        self.fps = fps
        self.robot_type = robot_type
        self.codebase_version = codebase_version
        self.task_name = task_name
        self.codec = codec
        self.splits = splits or {"train": "0:5"}
        # image_shape should be [height, width, channels]
        self.image_shape = tuple(image_shape) if image_shape is not None else (324, 576, 3)

    def _find_parquet_files(self) -> List[str]:
        """Return sorted list of parquet files under data/ (search chunk-* recursively)."""
        pattern = os.path.join(self.data_dir, "chunk-*", "**", "*.parquet")
        files = glob(pattern, recursive=True)
        files.sort()
        return files

    def _read_episode_lengths(self) -> List[int]:
        """
        Read each parquet file and return a list of frame counts.
        Each parquet file is treated as one episode (common LeRobot layout).
        """
        files = self._find_parquet_files()
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.data_dir}")
        lengths = []
        for p in files:
            try:
                df = pd.read_parquet(p)
                n = len(df)
                lengths.append(n)
                log.debug("Read %d rows from %s", n, p)
            except Exception as e:
                log.error("Failed to read parquet %s: %s", p, e)
                raise
        return lengths

    def generate_episodes_jsonl(self, output_path: Optional[str] = None) -> str:
        """
        Generate episodes.jsonl. Each parquet file becomes one episode.
        Returns path to written file.
        """
        lengths = self._read_episode_lengths()
        total_episodes = len(lengths)
        out = output_path or os.path.join(self.meta_dir, "episodes.jsonl")

        with open(out, "w", encoding="utf-8") as fh:
            for idx, ln in enumerate(lengths):
                entry = {"episode_index": idx, "tasks": [self.task_name], "length": ln}
                fh.write(json.dumps(entry) + "\n")
        log.info("Wrote episodes jsonl: %s (%d episodes)", out, total_episodes)

    def generate_info_json(self, output_path: Optional[str] = None) -> str:
        """
        Generate info.json using computed totals from parquet files.
        Returns path to written file.
        """
        lengths = self._read_episode_lengths()
        total_frames = int(sum(lengths))
        total_episodes = len(lengths)
        total_tasks = 1 if self.task_name else 0
        total_videos = total_episodes  # convention: one video per episode

        h, w, c = self.image_shape

        info = {
            "codebase_version": self.codebase_version,
            "robot_type": self.robot_type,
            "fps": int(self.fps),
            "total_episodes": int(total_episodes),
            "total_frames": int(total_frames),
            "total_tasks": int(total_tasks),
            "total_videos": int(total_videos),
            "splits": self.splits,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [14],
                    "names": [
                        "PSM1_joint_1",
                        "PSM1_joint_2",
                        "PSM1_joint_3",
                        "PSM1_joint_4",
                        "PSM1_joint_5",
                        "PSM1_joint_6",
                        "PSM1_jaw",
                        "PSM2_joint_1",
                        "PSM2_joint_2",
                        "PSM2_joint_3",
                        "PSM2_joint_4",
                        "PSM2_joint_5",
                        "PSM2_joint_6",
                        "PSM2_jaw",
                    ],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [14],
                    "names": [
                        "PSM1_joint_1",
                        "PSM1_joint_2",
                        "PSM1_joint_3",
                        "PSM1_joint_4",
                        "PSM1_joint_5",
                        "PSM1_joint_6",
                        "PSM1_jaw",
                        "PSM2_joint_1",
                        "PSM2_joint_2",
                        "PSM2_joint_3",
                        "PSM2_joint_4",
                        "PSM2_joint_5",
                        "PSM2_joint_6",
                        "PSM2_jaw",
                    ],
                },
                "observation.meta.tool_type": {
                    "dtype": "text",
                    "shape": [2],
                    "names": ["PSM1_tool_type", "PSM2_tool_type"],
                },
                "observation.images.camera_left": {
                    "dtype": "video",
                    "shape": [h, w, c],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.height": int(h),
                        "video.width": int(w),
                        "video.codec": self.codec,
                        "video.is_depth_map": False,
                        "video.fps": int(self.fps),
                        "video.channels": int(c),
                        "has_audio": False,
                    },
                },
                "observation.images.camera_right": {
                    "dtype": "video",
                    "shape": [h, w, c],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.height": int(h),
                        "video.width": int(w),
                        "video.codec": self.codec,
                        "video.is_depth_map": False,
                        "video.fps": int(self.fps),
                        "video.channels": int(c),
                        "has_audio": False,
                    },
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }

        out = output_path or os.path.join(self.meta_dir, "info.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=4)
            log.info("Wrote info json: %s", out)
        return out

    def generate_all(self) -> None:
        """Convenience: generate episodes.jsonl then info.json (uses same scan)."""
        self.generate_episodes_jsonl()
        self.generate_info_json()


if __name__ == "__main__":

    # Update these values as needed:
    dataset_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan LeRobot"  # root dataset directory containing data/ and meta/
    fps = 30
    task = "Needle Transfer"
    robot = "daVinci Surgical Robot"
    codebase = "v2.1"
    split_percent = {"train": "1:5", "val": "5:10"}  # format: name:range or just range
    image_shape = (540, 960, 3)
    codec = "h264"

    gen = MetaFilesGenerator(
        dataset_dir,
        fps=fps,
        robot_type=robot,
        codebase_version=codebase,
        task_name=task,
        splits=split_percent,
        image_shape=image_shape,
        codec=codec
    )
    gen.generate_all()
