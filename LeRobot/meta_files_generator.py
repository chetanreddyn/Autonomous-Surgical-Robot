#!/usr/bin/env python3
import os
import json
import logging
from glob import glob
from typing import List, Optional, Dict, Sequence, Tuple
import re

import pandas as pd
import numpy as np
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
        self.episode_data = self._read_episode_data()

    def _find_parquet_files(self) -> Dict[int, str]:
        """
        Return dict mapping episode_index -> parquet path.
        Extracts episode_index from filename pattern: episode_{episode_index:06d}.parquet
        """
        pattern = os.path.join(self.data_dir, "chunk-*", "**", "*.parquet")
        files = glob(pattern, recursive=True)
        
        episode_map = {}
        for fpath in files:
            basename = os.path.basename(fpath)
            # Match pattern: episode_000049.parquet
            match = re.match(r"episode_(\d+)\.parquet", basename)
            if match:
                episode_idx = int(match.group(1))
                episode_map[episode_idx] = fpath
        
        return episode_map
            
    def episode_stats_from_df(self, df_raw, episode_index):
        out = {"episode_index": int(episode_index), "stats": {}}

        if df_raw.empty:
            print("empty df")
        if "episode_index" in df_raw.columns:
            df = df_raw[df_raw["episode_index"] == episode_index]
            if df.empty:
                df_episode_index = df_raw["episode_index"].unique()
                print(f"Episode {episode_index}: empty after filtering, found episodes: {df_episode_index}")

        
        for col in df.columns:  
            s = df[col].dropna()
            if s.empty:
                continue
            
            # numeric scalar column
            if pd.api.types.is_numeric_dtype(s):
                vals = s.values.astype(float)
                out["stats"][col] = {
                    "min": [float(vals.min())],
                    "max": [float(vals.max())],
                    "mean": [float(vals.mean())],
                    "std": [float(vals.std(ddof=0))],
                    "count": [len(vals)]
                }
                continue
            
            # try array-like column (observation.state, action, etc)
            try:
                arrs = [np.asarray(v, dtype=float) for v in s]
                stacked = np.stack(arrs, axis=0)  # shape: (N, ...)
                
                out["stats"][col] = {
                    "min": stacked.min(axis=0).tolist(),
                    "max": stacked.max(axis=0).tolist(),
                    "mean": stacked.mean(axis=0).tolist(),
                    "std": stacked.std(axis=0, ddof=0).tolist(),
                    "count": [len(stacked)]
                }
            except (ValueError, TypeError):
                # skip non-numeric columns (strings, mixed types, etc)
                continue
        
        return out

    def _read_episode_data(self) -> Dict[int, Tuple[str, int, Dict]]:
        """
        Read each parquet file and return dict mapping episode_index -> (path, frame_count, stats_dict).
        """
        episode_map = self._find_parquet_files()
        if not episode_map:
            raise FileNotFoundError(f"No parquet files found under {self.data_dir}")
        episode_data = {}
        for episode_idx in sorted(episode_map.keys()):
            fpath = episode_map[episode_idx]

            try:
                df = pd.read_parquet(fpath)
                n = len(df)
                stats = self.episode_stats_from_df(df, episode_idx)
                episode_data[episode_idx] = (fpath, n, stats)
            except Exception as e:
                log.error("Failed to read parquet %s: %s", fpath, e)
                raise
        return episode_data

    def generate_tasks_jsonl(self, tasks: Dict[int, str], output_path: Optional[str] = None) -> str:
        """
        Generate tasks.jsonl with provided task descriptions.
        
        Args:
            tasks: Dict mapping task_index -> task description string
            output_path: Optional custom output path
            
        Returns:
            Path to written file
        """
        out = output_path or os.path.join(self.meta_dir, "tasks.jsonl")
        
        with open(out, "w", encoding="utf-8") as fh:
            for task_idx in sorted(tasks.keys()):
                entry = {"task_index": task_idx, "task": tasks[task_idx]}
                fh.write(json.dumps(entry) + "\n")
        
        log.info("Wrote tasks jsonl: %s (%d tasks)", out, len(tasks))
        return out

    def generate_episodes_stats_jsonl(self, output_path: Optional[str] = None) -> str:
        """
        Generate episodes_stats.jsonl with statistics for each episode.
        Returns path to written file.
        """
        out = output_path or os.path.join(self.meta_dir, "episodes_stats.jsonl")
        
        with open(out, "w", encoding="utf-8") as fh:
            for episode_idx in sorted(self.episode_data.keys()):
                _, _, stats = self.episode_data[episode_idx]
                fh.write(json.dumps(stats) + "\n")

        log.info("Wrote episodes stats jsonl: %s (%d episodes)", out, len(self.episode_data))
        return out

    def generate_episodes_jsonl(self, output_path: Optional[str] = None) -> str:
        """
        Generate episodes.jsonl. Each parquet file becomes one episode.
        Returns path to written file.
        """
        # episode_data = self._read_episode_data()
        out = output_path or os.path.join(self.meta_dir, "episodes.jsonl")

        with open(out, "w", encoding="utf-8") as fh:
            for episode_idx in sorted(self.episode_data.keys()):
                _, length, _ = self.episode_data[episode_idx]
                entry = {"episode_index": episode_idx, "tasks": [self.task_name], "length": length}
                fh.write(json.dumps(entry) + "\n")
        log.info("Wrote episodes jsonl: %s (%d episodes)", out, len(self.episode_data))
        return out

    def generate_info_json(self, output_path: Optional[str] = None) -> str:
        """
        Generate info.json using computed totals from parquet files.
        Returns path to written file.
        """
        # episode_data = self._read_episode_data()
        total_frames = sum(length for _, length, _ in self.episode_data.values())
        total_episodes = len(self.episode_data)
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

    def generate_all(self, tasks: Optional[Dict[int, str]] = None) -> None:
        """Convenience: generate all meta files."""
        self.generate_episodes_jsonl()
        self.generate_episodes_stats_jsonl()
        self.generate_info_json()
        if tasks:
            self.generate_tasks_jsonl(tasks)


if __name__ == "__main__":

    # Update these values as needed:
    dataset_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan LeRobot"  # root dataset directory containing data/ and meta/
    fps = 30
    task = "Needle Transfer"
    robot = "daVinci Surgical Robot"
    codebase = "v2.1"
    split_percent = {"train": "0:70", "val": "71:85", "test": "86:94"}  # format: name:range or just range
    image_shape = (540, 960, 3)
    codec = "h264"
    tasks = {0:"Needle Transfer"}

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

    gen.generate_all(tasks)