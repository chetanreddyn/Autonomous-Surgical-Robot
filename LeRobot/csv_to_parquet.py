import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import json

class CSVtoParquetConverter:
    """
    Converts CSV robot recordings to LeRobot Parquet format.
    Processes multiple demo subfolders under a single experiment directory.
    Saves one Parquet file per CSV as episode_0000XX.parquet in the same folder.
    """

    def __init__(self, exp_dir, task_index=0, tool_type="FENESTRATED_BIPOLAR_FORCEPS:420205", start_index=0):
        self.exp_dir = exp_dir
        self.task_index = task_index
        self.global_index = start_index
        self.num_arms = 2  # assuming two arms: PSM1 and PSM2
        self.observation_cols = [
                'PSM1_joint_1','PSM1_joint_2','PSM1_joint_3','PSM1_joint_4',
                'PSM1_joint_5','PSM1_joint_6','PSM1_jaw',
                'PSM2_joint_1','PSM2_joint_2','PSM2_joint_3','PSM2_joint_4',
                'PSM2_joint_5','PSM2_joint_6','PSM2_jaw'
            ]
        self.action_cols = [
                'PSM1_joint_1','PSM1_joint_2','PSM1_joint_3','PSM1_joint_4',
                'PSM1_joint_5','PSM1_joint_6','PSM1_jaw',
                'PSM2_joint_1','PSM2_joint_2','PSM2_joint_3','PSM2_joint_4',
                'PSM2_joint_5','PSM2_joint_6','PSM2_jaw'
            ]

    def _load_tools_for_demo(self, demo_path, config_filename="experiment_config.json"):
        """
        Try to load tools_used from demo_path/experiment_config.json first.
        If not present, try exp_dir/experiment_config.json.
        Return a list suitable for self.tool_type or None if not found.
        """
        p = Path(demo_path) / config_filename
        try:
            with open(p, "r") as f:
                meta = json.load(f)
            tools = meta.get("tools_used")
            if tools and isinstance(tools, list):
                # return as-is; if user expects two entries, caller can handle
                return tools[:self.num_arms]
        except Exception as e:
            print(f"Failed to read meta file {p}: {e}")
        return None

    def _process_episode(self, csv_path, episode_index):
        """
        Converts a single CSV episode to a list of dicts in target structure.
        """
        df = pd.read_csv(csv_path)
        df['timestamp'] = df['Time (Seconds)'] - df['Time (Seconds)'].iloc[0]

        rows = []
        n_steps = len(df)

        for i in range(n_steps - 1):
            # Flattened observation.state: PSM1 joints followed by PSM2 joints (7+7=14)
            state = df.loc[i, self.observation_cols].tolist()

            # Flattened action: next timestep's same 14 joint values
            action = df.loc[i+1, self.action_cols].tolist()

            row = {
                "observation.state": state,  # flattened 14 values
                "action": action,            # flattened 14 values
                "observation.meta.tool_type": self.tool_type,
                "timestamp": float(df.loc[i, 'timestamp']),
                "frame_index": int(df.loc[i, 'Frame Number']),
                "episode_index": episode_index,
                "index": self.global_index,
                "task_index": self.task_index
            }

            rows.append(row)
            self.global_index += 1

        return rows

    def convert(self, *, demo_start, demo_end):
        """
        Processes CSV files in demo subfolders sequentially (Demo1, Demo2, ...).
        Saves one Parquet file per CSV as episode_0000XX.parquet in the same folder.
        """

        episode_idx = 0
        for demo in range(demo_start, demo_end + 1):
            demo_path = os.path.join(self.exp_dir, f"Demo{demo}")
            csv_path = os.path.join(demo_path, "data.csv")
            self.tool_type = self._load_tools_for_demo(demo_path)

            if os.path.isfile(csv_path):
                episode_rows = self._process_episode(csv_path, episode_idx)

                # Save Parquet in the same folder with zero-padded episode index
                parquet_file = os.path.join(demo_path, f"episode_{episode_idx:06d}.parquet")
                table = pa.Table.from_pylist(episode_rows)
                pq.write_table(table, parquet_file)

                print(f"Saved {len(episode_rows)} steps to {parquet_file}")
                episode_idx += 1
            else:
                print(f"No data.csv found in {demo_path}, skipping.")

    def clean_dir(self, pattern: str = "*.parquet", do_delete: bool = False):
        """
        Find and optionally delete parquet files under exp_dir.
        
        Args:
            pattern: glob pattern for files to find (default: *.parquet)
            do_delete: if True, actually delete files; otherwise just list them
        
        Returns:
            List of found/deleted Path objects
        """
        root = Path(self.exp_dir)
        files = list(root.rglob(pattern))
        
        if not files:
            print(f"No files found matching pattern: {pattern}")
            return []
        
        if do_delete:
            deleted = []
            for f in files:
                try:
                    f.unlink()
                    deleted.append(f)
                    print(f"Deleted: {f}")
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")
            print(f"Deleted {len(deleted)} parquet files.")
        else:
            print(f"Found {len(files)} parquet files (dry-run, not deleting):")
            # for f in files:
            #     print(f"  {f}")

# Usage
if __name__ == "__main__":
    demo_start = 1  # Update as needed
    demo_end = 20   # Update as needed
    exp_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer"
    clean_dir = False  # Set to True to delete parquet files

    converter = CSVtoParquetConverter(
        exp_dir=exp_dir,
    )
    # Clean parquet files (dry-run)

    if clean_dir:
        converter.clean_dir(do_delete=False)
        input("Press Enter to confirm deletion of these files...")
        converter.clean_dir(do_delete=True)

    converter.convert(demo_start=demo_start, demo_end=demo_end)