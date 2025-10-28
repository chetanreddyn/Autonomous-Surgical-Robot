import argparse
import os
import sys

from images_to_videos import VideoGenerator
from video_format_corrector import VideoConverter
from csv_to_parquet import CSVtoParquetConverter

def main():
    parser = argparse.ArgumentParser(description="LeRobot utilities: images->videos, re-encode videos, CSV->Parquet")
    parser.add_argument("--exp-dir", type=str,
                        default="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan")
    parser.add_argument("--demo-start", type=int, default=1)
    parser.add_argument("--demo-end", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-codec", type=str, default="mp4v") # Initial codec for images->videos step
    parser.add_argument("--crf", type=int, default=30)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    demo_start = args.demo_start
    demo_end = args.demo_end

    # 1) Generate videos from images
    print("\n=== Generating videos from images ===")
    try:
        vg = VideoGenerator(fps=args.fps, codec=args.video_codec)
        vg.generate_videos_for_all_demos(exp_dir, demo_start, demo_end)
    except Exception as e:
        print(f"[main] images->videos step failed: {e}", file=sys.stderr)

    # 2) Re-encode videos to H.264
    print("\n=== Re-encoding videos to H.264 ===")
    try:
        vc = VideoConverter(base_dir=exp_dir, demo_start=demo_start, demo_end=demo_end, crf=args.crf)
        vc.convert_all()
        vc.replace_originals()
    except Exception as e:
        print(f"[main] video re-encode step failed: {e}", file=sys.stderr)

    # 3) Convert CSV -> Parquet
    print("\n=== Converting CSV to Parquet ===")
    try:
        converter = CSVtoParquetConverter(exp_dir=exp_dir)
        converter.convert(demo_start=demo_start, demo_end=demo_end)
    except Exception as e:
        print(f"[main] csv->parquet step failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()