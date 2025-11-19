import os
import shutil
import pdb

class LeRobotDatasetGenerator:
    """
    Generates a new dataset with the same structure as dummy_lerobot_dataset,
    copying only parquet files and videos (ignoring images) from the Collaborative folder.
    All files are renamed to episode_xxxxxx.*
    """

    def __init__(self, src_root, dst_root):
        self.src_root = src_root
        self.dst_root = dst_root

    def create_structure(self):
        """Creates the folder structure for the new dataset."""
        os.makedirs(os.path.join(self.dst_root, "data/chunk-000"), exist_ok=True)
        os.makedirs(os.path.join(self.dst_root, "videos/chunk-000/observation.images.camera_left"), exist_ok=True)
        os.makedirs(os.path.join(self.dst_root, "videos/chunk-000/observation.images.camera_right"), exist_ok=True)
        os.makedirs(os.path.join(self.dst_root, "meta"), exist_ok=True)

    def copy_parquet_files(self):
        """Copies and renames episode parquet files from each demo to data/chunk-000/ preserving episode index."""
        demo_dirs = sorted([d for d in os.listdir(self.src_root) if d.startswith("Demo")])
        
        for demo in demo_dirs:
            demo_path = os.path.join(self.src_root, demo)
            parquet_files = sorted([f for f in os.listdir(demo_path) if f.endswith(".parquet")])
            
            for file in parquet_files:
                src_file = os.path.join(demo_path, file)
                # Keep the same filename (episode_XXXXXX.parquet)
                dst_file = os.path.join(self.dst_root, "data/chunk-000", file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {demo}/{file} -> data/chunk-000/{file}")

    def copy_videos(self):
        """
        Copies and renames videos from each demo to videos/chunk-000/camera_key/ preserving episode index.
        """
        demo_dirs = sorted([d for d in os.listdir(self.src_root) if d.startswith("Demo")])
        
        for demo in demo_dirs:
            videos_dir = os.path.join(self.src_root, demo, "videos")
            if not os.path.exists(videos_dir):
                print(f"Warning: Videos not found in {videos_dir}, skipping")
                continue
            
            # Extract episode number from parquet file in this demo
            demo_path = os.path.join(self.src_root, demo)
            parquet_files = [f for f in os.listdir(demo_path) if f.endswith(".parquet")]
            if not parquet_files:
                print(f"Warning: No parquet file in {demo}, skipping videos")
                continue
            
            # Use the parquet filename to determine episode index
            episode_filename = parquet_files[0]  # e.g., "episode_000005.parquet"
            
            for camera_key in ["camera_left", "camera_right"]:
                dst_camera_dir = os.path.join(self.dst_root, "videos/chunk-000", "observation.images." + camera_key)
                video_file = os.path.join(videos_dir, camera_key + ".mp4")
                
                if os.path.exists(video_file):
                    # Replace .parquet with .mp4 in the episode filename
                    video_dst_name = episode_filename.replace(".parquet", ".mp4")
                    dst_file = os.path.join(dst_camera_dir, video_dst_name)
                    shutil.copy2(video_file, dst_file)
                    print(f"Copied {demo}/videos/{camera_key}.mp4 -> {video_dst_name}")


    def copy_readme(self):
        """Copies README.md from dummy_lerobot_dataset if present."""
        src_readme = os.path.join(os.path.dirname(self.src_root), "dummy_lerobot_dataset", "README.md")
        dst_readme = os.path.join(self.dst_root, "README.md")
        if os.path.exists(src_readme):
            shutil.copy2(src_readme, dst_readme)

    def run(self):
        """Runs the full dataset generation process."""
        self.create_structure()
        self.copy_parquet_files()
        self.copy_videos()
        print(f"Dataset generated at {self.dst_root}")

# Usage
if __name__ == "__main__":
    src_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan"
    dst_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan LeRobot"
    # src_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Tissue Retraction Chetan"
    # dst_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Tissue Retraction Chetan LeRobot"
    generator = LeRobotDatasetGenerator(src_root, dst_root)
    generator.run()