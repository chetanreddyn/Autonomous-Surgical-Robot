import argparse
import os
import sys

from images_to_videos import VideoGenerator
from video_format_corrector import VideoConverter
from csv_to_parquet import CSVtoParquetConverter

def main():
    parser = argparse.ArgumentParser(description="LeRobot utilities: images->videos, re-encode videos, CSV->Parquet")
    parser.add_argument("--exp-dir", type=str,
                        default="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Peg Transfer Chetan")
    parser.add_argument("-s", "--demo-start", type=int, default=1)
    parser.add_argument("-e", "--demo-end", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-codec", type=str, default="mp4v") # Initial codec for images->videos step
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--task-index", type=int, default=0, help="Task index for parquet conversion (0=needle transfer, 1=tissue retraction, 2=Peg Transfer)")
    parser.add_argument("-v", "--videos", action="store_true", help="Only generate and re-encode videos")
    parser.add_argument("-p", "--parquet", action="store_true", help="Only convert CSV to Parquet")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    demo_start = args.demo_start
    demo_end = args.demo_end
    task_index = args.task_index

    # Decide which steps to run:
    # If neither -v nor -p passed -> run all steps (legacy behavior).
    run_videos = args.videos 
    run_parquet = args.parquet
    if run_videos:
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

    if run_parquet:
        # 3) Convert CSV -> Parquet
        print("\n=== Converting CSV to Parquet ===")
        try:
            converter = CSVtoParquetConverter(exp_dir=exp_dir, task_index=task_index)
            converter.convert(demo_start=demo_start, demo_end=demo_end)
        except Exception as e:
            print(f"[main] csv->parquet step failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()