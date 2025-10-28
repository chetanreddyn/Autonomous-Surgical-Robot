import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

class NpyToImageConverter:
    """
    Convert .npy files to images.

    Usage:
      conv = NpyToImageConverter()
      conv.convert_demo_range(root_dir, start, end, prefix="Demo", images_subfolder="images")
    """
    def __init__(self):
        self.ext = ".png"
        pass

    def convert_file(self, npy_path: Path, output_dir: Optional[Path] = None) -> None:
        try:
            arr = np.load(npy_path)
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}")
            return

        img = arr.astype(np.uint8)
        out_dir = output_dir or npy_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{npy_path.stem}{self.ext}"
        success = cv2.imwrite(str(out_path), img)
        if not success:
            print(f"Failed to save image: {out_path}")

    def convert_directory(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        inp = Path(input_dir)
        out = Path(output_dir) if output_dir else inp
        if not inp.exists():
            print(f"Input directory does not exist: {inp}")
            return
        npy_files = sorted(inp.glob("*.npy"))
        if not npy_files:
            print(f"No .npy files in {inp}")
            return
        print(f"Converting {len(npy_files)} .npy files in {inp} -> {out}")
        for p in npy_files:
            self.convert_file(p, out)

    def convert_demo_range(
        self,
        root_dir: str,
        start: int,
        end: int,
        prefix: str = "Demo",
        images_subfolder: str = "images",
        output_to_images_subfolder: bool = True
    ) -> None:
        root = Path(root_dir)
        if start > end:
            print("Start index greater than end index; nothing to do.")
            return

        for idx in range(start, end + 1):
            demo_folder = root / f"{prefix}{idx}"
            img_folder = demo_folder / images_subfolder
            if img_folder.exists() and any(img_folder.glob("*.npy")):
                input_folder = img_folder
                output_folder = img_folder if output_to_images_subfolder else demo_folder
            elif demo_folder.exists() and any(demo_folder.glob("*.npy")):
                input_folder = demo_folder
                output_folder = demo_folder
            else:
                print(f"Skipping missing or empty demo: {demo_folder}")
                continue

            print(f"Converting demo {prefix}{idx}: {input_folder} -> {output_folder}")
            self.convert_directory(str(input_folder), str(output_folder))


if __name__ == "__main__":
    LOGGING_FOLDER = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Needle Transfer Chetan/"
    demo_start = 1
    demo_end = 1

    converter = NpyToImageConverter()
    converter.convert_demo_range(LOGGING_FOLDER, demo_start, demo_end, prefix="Demo", images_subfolder="images")