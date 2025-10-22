import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

class LeRobotDatasetConverter:
    """
    Converts CSV robot recordings to LeRobot Parquet format.
    Processes multiple demo subfolders under a single experiment directory.
    Saves one Parquet file per CSV as episode_0000XX.parquet in the same folder.
    """

    def __init__(self, exp_dir, task_index=0, tool_type="FENESTRATED_BIPOLAR_FORCEPS:420205", start_index=0):
        self.exp_dir = exp_dir
        self.task_index = task_index
        self.tool_type = [tool_type, tool_type]
        self.global_index = start_index
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

    def convert(self):
        """
        Processes CSV files in demo subfolders sequentially (Demo1, Demo2, ...).
        Saves one Parquet file per CSV as episode_0000XX.parquet in the same folder.
        """
        all_entries = os.listdir(self.exp_dir)
        demo_folders = sorted([d for d in all_entries if d.startswith("Demo")])

        episode_idx = 0
        for demo in demo_folders:
            demo_path = os.path.join(self.exp_dir, demo)
            csv_path = os.path.join(demo_path, "data.csv")

            if os.path.isfile(csv_path):
                episode_rows = self._process_episode(csv_path, episode_idx)

                # Save Parquet in the same folder with zero-padded episode index
                parquet_file = os.path.join(demo_path, f"episode_{episode_idx:06d}.parquet")
                table = pa.Table.from_pylist(episode_rows)
                pq.write_table(table, parquet_file)

                print(f"Saved {len(episode_rows)} steps from {csv_path} → {parquet_file}")
                episode_idx += 1
            else:
                print(f"No data.csv found in {demo_path}, skipping.")


# Usage
converter = LeRobotDatasetConverter(
    exp_dir="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer",
)
converter.convert()
