import os
import subprocess
from glob import glob

class VideoConverter:
    def __init__(self, base_dir, demo_start=1, demo_end=5, crf=30):
        """
        Initialize the VideoConverter object.
        
        Args:
            base_dir (str): Base directory containing Demo folders
            demo_start (int): Starting demo number
            demo_end (int): Ending demo number
            crf (int): Constant Rate Factor for H.264 compression (lower is better quality)
        """
        self.base_dir = base_dir
        self.demo_start = demo_start
        self.demo_end = demo_end
        self.crf = crf

    def convert_all(self):
        """Convert all videos in the demo folders to H.264 format."""
        # print("=== Converting videos to H.264 ===")
        for demo_num in range(self.demo_start, self.demo_end + 1):
            videos_dir = self._get_videos_dir(demo_num)
            if not os.path.exists(videos_dir):
                # print(f"Videos directory not found: {videos_dir}")
                continue

            print(f"--- Processing Demo{demo_num} ---")
            video_files = glob(os.path.join(videos_dir, "*.mp4"))

            for video_file in video_files:
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                output_file = os.path.join(videos_dir, f"{base_name}_h264.mp4")
                self._convert_to_h264(video_file, output_file)

    def replace_originals(self):
        """Replace the original videos with the converted H.264 versions."""
        print("\n=== Replacing original videos ===")
        for demo_num in range(self.demo_start, self.demo_end + 1):
            videos_dir = self._get_videos_dir(demo_num)
            if not os.path.exists(videos_dir):
                continue

            # print(f"--- Replacing videos in Demo{demo_num} ---")
            h264_files = glob(os.path.join(videos_dir, "*_h264.mp4"))

            for h264_file in h264_files:
                original_file = h264_file.replace("_h264.mp4", ".mp4")
                if os.path.exists(original_file):
                    os.remove(original_file)
                    # print(f"Removed: {original_file}")
                os.rename(h264_file, original_file)
                # print(f"Renamed: {h264_file} -> {original_file}")

    def _convert_to_h264(self, input_path, output_path):
        """Internal method to convert a single video using ffmpeg."""
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(self.crf),
            '-b:v', '0',
            '-y',
            output_path
        ]
        try:
            # print(f"Converting {input_path} -> {output_path}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # print(f"✓ Converted: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error converting {input_path}")
            print(f"  stderr: {e.stderr}")

    def _get_videos_dir(self, demo_num):
        """Get the path to the videos directory for a given demo number."""
        return os.path.join(self.base_dir, f"Demo{demo_num}", "videos")


if __name__ == "__main__":
    base_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer"
    demo_start = 1
    demo_end = 20

    converter = VideoConverter(base_dir, demo_start=demo_start, demo_end=demo_end, crf=30)
    converter.convert_all()
    converter.replace_originals()
    print("\n=== Conversion complete! ===")
