#!/usr/bin/env python

import os
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_video_from_image_folder(image_folder, output_path, fps=30, resolution=None):
    """
    Create a video from a folder of images.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path for output video file
        fps: Frames per second for output video
        resolution: Tuple (width, height) for output resolution, None to use image size
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get all image files (support multiple formats)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        # Sort by filename (assuming they're numbered)
        image_files = sorted(image_files, key=lambda x: x.name)
        
        if not image_files:
            logger.warning(f"No images found in {image_folder}")
            return False
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            logger.error(f"Could not read first image: {image_files[0]}")
            return False
        
        height, width = first_img.shape[:2]
        
        # Use provided resolution or image resolution
        if resolution:
            width, height = resolution
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process all images
        for i, img_path in enumerate(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            
            # Resize if needed
            if resolution and (img.shape[1], img.shape[0]) != resolution:
                img = cv2.resize(img, resolution)
            
            out.write(img)
            
            if i % 100 == 0:
                logger.debug(f"Processed {i}/{len(image_files)} images")
        
        out.release()
        logger.info(f"Video created successfully: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return False

def concatenate_videos_side_by_side(left_video_path, right_video_path, output_path):
    """
    Concatenate two videos side by side.
    
    Args:
        left_video_path: Path to left video
        right_video_path: Path to right video
        output_path: Path for output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open both videos
        cap_left = cv2.VideoCapture(left_video_path)
        cap_right = cv2.VideoCapture(right_video_path)
        
        # Get video properties
        fps = int(cap_left.get(cv2.CAP_PROP_FPS))
        width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output dimensions
        out_width = width_left + width_right
        out_height = max(height_left, height_right)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                break
            
            # Create combined frame
            combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            combined[:height_left, :width_left] = frame_left
            combined[:height_right, width_left:width_left+width_right] = frame_right
            
            out.write(combined)
        
        cap_left.release()
        cap_right.release()
        out.release()
        
        logger.info(f"Concatenated video created: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        return False