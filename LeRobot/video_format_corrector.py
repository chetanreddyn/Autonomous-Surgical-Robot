import os
import subprocess
from glob import glob

def convert_video_to_h264(input_path, output_path, crf=30):
    """
    Convert a video to H.264 using ffmpeg.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        crf (int): Constant Rate Factor (18-28 for good quality, 30+ for smaller files)
    """
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-b:v', '0',
        '-y',  # Overwrite output file if it exists
        output_path
    ]
    
    try:
        print(f"Converting {input_path} -> {output_path}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Converted: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting {input_path}: {e}")
        print(f"  stderr: {e.stderr}")

def convert_all_videos_in_folder(base_dir, demo_start=1, demo_end=5, crf=30):
    """
    Convert all videos in Demo folders to H.264.
    
    Args:
        base_dir (str): Base directory containing Demo folders
        demo_start (int): Starting demo number
        demo_end (int): Ending demo number
        crf (int): Quality setting for H.264 encoding
    """
    for demo_num in range(demo_start, demo_end + 1):
        demo_dir = os.path.join(base_dir, f"Demo{demo_num}")
        videos_dir = os.path.join(demo_dir, "videos")
        
        if not os.path.exists(videos_dir):
            print(f"Videos directory not found: {videos_dir}")
            continue
            
        print(f"\n--- Processing Demo{demo_num} ---")
        
        # Find all .mp4 files in videos directory
        video_files = glob(os.path.join(videos_dir, "*.mp4"))
        
        for video_file in video_files:
            # Create output filename with _h264 suffix
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            output_file = os.path.join(videos_dir, f"{base_name}_h264.mp4")
            
            convert_video_to_h264(video_file, output_file, crf=crf)

def replace_original_videos(base_dir, demo_start=1, demo_end=5):
    """
    Replace original videos with H.264 versions and clean up.
    
    Args:
        base_dir (str): Base directory containing Demo folders
        demo_start (int): Starting demo number
        demo_end (int): Ending demo number
    """
    for demo_num in range(demo_start, demo_end + 1):
        demo_dir = os.path.join(base_dir, f"Demo{demo_num}")
        videos_dir = os.path.join(demo_dir, "videos")
        
        if not os.path.exists(videos_dir):
            continue
            
        print(f"\n--- Replacing videos in Demo{demo_num} ---")
        
        # Find all _h264.mp4 files
        h264_files = glob(os.path.join(videos_dir, "*_h264.mp4"))
        
        for h264_file in h264_files:
            # Get original filename
            base_name = os.path.basename(h264_file).replace("_h264.mp4", ".mp4")
            original_file = os.path.join(videos_dir, base_name)
            
            if os.path.exists(original_file):
                # Remove original
                os.remove(original_file)
                print(f"Removed: {original_file}")
                
                # Rename h264 version to original name
                os.rename(h264_file, original_file)
                print(f"Renamed: {h264_file} -> {original_file}")

if __name__ == "__main__":
    base_dir = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer"
    
    # Step 1: Convert all videos to H.264
    print("=== Converting videos to H.264 ===")
    convert_all_videos_in_folder(base_dir, demo_start=1, demo_end=10, crf=30)
    
    # Step 2: Replace original videos with H.264 versions
    print("\n=== Replacing original videos ===")
    replace_original_videos(base_dir, demo_start=1, demo_end=10)
    
    print("\n=== Conversion complete! ===")