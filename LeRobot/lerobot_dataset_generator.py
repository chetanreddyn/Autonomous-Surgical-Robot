import os
import shutil
import pdb

class DummyObjectTransferVideoGenerator:
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
        """Copies and renames episode parquet files from each demo to data/chunk-000/ as episode_xxxxxx.parquet."""
        demo_dirs = sorted([d for d in os.listdir(self.src_root) if d.startswith("Demo")])
        episode_idx = 0
        for demo in demo_dirs:
            demo_path = os.path.join(self.src_root, demo)
            for file in sorted(os.listdir(demo_path)):
                if file.endswith(".parquet"):
                    src_file = os.path.join(demo_path, file)
                    dst_file = os.path.join(self.dst_root, "data/chunk-000", f"episode_{episode_idx:06d}.parquet")
                    shutil.copy2(src_file, dst_file)
                    episode_idx += 1

    def copy_videos(self):
        """
        Copies and renames videos from each demo to videos/chunk-000/camera_key/ as episode_xxxxxx.mp4.
        Assumes each demo/videos/ contains subfolders for each camera key.
        """
        demo_dirs = sorted([d for d in os.listdir(self.src_root) if d.startswith("Demo")])
        episode_idx = 0
        for demo in demo_dirs:
            videos_dir = os.path.join(self.src_root, demo, "videos")
            if not os.path.exists(videos_dir):
                raise Exception(f"Videos not found in {videos_dir}")
            
            for camera_key in ["camera_left", "camera_right"]:
                dst_camera_dir = os.path.join(self.dst_root, "videos/chunk-000", "observation.images." + camera_key)
                video_file = os.path.join(videos_dir,camera_key+".mp4")
                if video_file.endswith(".mp4"):
                    dst_file = os.path.join(dst_camera_dir, f"episode_{episode_idx:06d}.mp4")
                    shutil.copy2(video_file, dst_file)
            
            episode_idx += 1

    def copy_meta(self):
        """Copies meta files from dummy_lerobot_dataset/meta to new meta folder."""
        src_meta_dir = os.path.join(os.path.dirname(self.src_root), "dummy_lerobot_dataset", "meta")
        dst_meta_dir = os.path.join(self.dst_root, "meta")
        if os.path.exists(src_meta_dir):
            for file in os.listdir(src_meta_dir):
                src_file = os.path.join(src_meta_dir, file)
                dst_file = os.path.join(dst_meta_dir, file)
                shutil.copy2(src_file, dst_file)

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
        # self.copy_meta()
        # self.copy_readme()
        print(f"Dataset generated at {self.dst_root}")

# Usage
if __name__ == "__main__":
    src_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer"
    dst_root = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer LeRobot Format"
    generator = DummyObjectTransferVideoGenerator(src_root, dst_root)
    generator.run()