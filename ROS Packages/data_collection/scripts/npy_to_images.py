import os
import numpy as np
import cv2
from pathlib import Path

class NpyToImageConverter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_all(self):
        npy_files = list(self.input_dir.glob("*.npy"))
        print(f"Found {len(npy_files)} .npy files.")
        
        for i,npy_file in enumerate(npy_files):
            # print(f"Converting file {i+1}/{len(npy_files)}")
            self.convert_file(npy_file)

    def convert_file(self, npy_file: Path):
        try:
            arr = np.load(npy_file)
        except Exception as e:
            print(f"Failed to load {npy_file.name}: {e}")
            return

        # Normalize array to 0-255 if needed
        arr = self._normalize_array(arr)

        # Convert to uint8 for image saving
        image = arr.astype(np.uint8)

        # Save image
        output_path = self.output_dir / f"{npy_file.stem}.png"
        success = cv2.imwrite(str(output_path), image)
        if not success:
            print(f"Failed to save: {output_path}")

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        if np.issubdtype(arr.dtype, np.floating) or arr.max() > 255:
            arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        return arr
    
def convert_demo_range(root_dir: str, start: int, end: int, prefix: str = "Demo", images_subfolder: str = "images"):
    """
    Convenience method: convert .npy files for demo folders named <prefix><i> for i in [start..end].
    Each demo is searched for an 'images' subfolder first, then the demo folder itself.
    Example: convert_demo_range("/path/to/root", 1, 5, prefix="Demo")
    """
    root = Path(root_dir)
    if start > end:
        print("Start index is greater than end index; nothing to do.")
        return

    for idx in range(start, end + 1):
        demo_folder = root / f"{prefix}{idx}"
        input_folder = demo_folder / images_subfolder
        if not input_folder.exists():
            # fallback to demo folder itself
            if demo_folder.exists() and any(demo_folder.glob("*.npy")):
                input_folder = demo_folder
            else:
                print(f"Skipping missing demo folder or no npy files: {demo_folder}")
                continue

        print(f"Converting demo {prefix}{idx} -> {input_folder}")
        conv = NpyToImageConverter(str(input_folder), str(input_folder))
        conv.convert_all()

if __name__ == "__main__":
    LOGGING_FOLDER = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Two Handed Needle Transfer/"


    demo_start = 1
    demo_end = 10

    convert_demo_range(LOGGING_FOLDER, demo_start, demo_end, prefix="Demo", images_subfolder="images")