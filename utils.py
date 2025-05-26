"""
Utility functions for the Vehicle Detection System
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Union
import json
import logging

logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory (str): Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        dict: Video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }

    cap.release()
    return info

def save_detections_json(detections: List[dict], output_path: str) -> None:
    """
    Save detection results to a JSON file.

    Args:
        detections (List[dict]): List of detection results
        output_path (str): Path to save the JSON file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
        logger.info(f"Detections saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving detections: {e}")

def load_detections_json(json_path: str) -> List[dict]:
    """
    Load detection results from a JSON file.

    Args:
        json_path (str): Path to the JSON file

    Returns:
        List[dict]: List of detection results
    """
    try:
        with open(json_path, 'r') as f:
            detections = json.load(f)
        logger.info(f"Detections loaded from: {json_path}")
        return detections
    except Exception as e:
        logger.error(f"Error loading detections: {e}")
        return []

def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    Resize an image to target size.

    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target (width, height)
        keep_aspect_ratio (bool): Whether to maintain aspect ratio

    Returns:
        np.ndarray: Resized image
    """
    if keep_aspect_ratio:
        height, width = image.shape[:2]
        target_width, target_height = target_size

        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height))

        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        return canvas
    else:
        return cv2.resize(image, target_size)

def crop_vehicle_region(image: np.ndarray, bbox: dict, padding: int = 10) -> np.ndarray:
    """
    Crop vehicle region from image with optional padding.

    Args:
        image (np.ndarray): Input image
        bbox (dict): Bounding box dictionary with x1, y1, x2, y2
        padding (int): Padding around the bounding box

    Returns:
        np.ndarray: Cropped vehicle region
    """
    height, width = image.shape[:2]

    # Add padding
    x1 = max(0, bbox['x1'] - padding)
    y1 = max(0, bbox['y1'] - padding)
    x2 = min(width, bbox['x2'] + padding)
    y2 = min(height, bbox['y2'] + padding)

    return image[y1:y2, x1:x2]

def calculate_iou(box1: dict, box2: dict) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (dict): First bounding box
        box2 (dict): Second bounding box

    Returns:
        float: IoU value
    """
    # Calculate intersection area
    x1_inter = max(box1['x1'], box2['x1'])
    y1_inter = max(box1['y1'], box2['y1'])
    x2_inter = min(box1['x2'], box2['x2'])
    y2_inter = min(box1['y2'], box2['y2'])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union area
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def filter_detections_by_size(detections: List[dict], min_width: int = 30,
                             min_height: int = 30, max_width: int = None,
                             max_height: int = None) -> List[dict]:
    """
    Filter detections by bounding box size.

    Args:
        detections (List[dict]): List of detections
        min_width (int): Minimum width
        min_height (int): Minimum height
        max_width (int): Maximum width (None for no limit)
        max_height (int): Maximum height (None for no limit)

    Returns:
        List[dict]: Filtered detections
    """
    filtered = []

    for detection in detections:
        bbox = detection['bbox']
        width = bbox['width']
        height = bbox['height']

        # Check minimum size
        if width < min_width or height < min_height:
            continue

        # Check maximum size
        if max_width and width > max_width:
            continue
        if max_height and height > max_height:
            continue

        filtered.append(detection)

    return filtered

def merge_overlapping_detections(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """
    Merge overlapping detections using IoU threshold.

    Args:
        detections (List[dict]): List of detections
        iou_threshold (float): IoU threshold for merging

    Returns:
        List[dict]: Merged detections
    """
    if not detections:
        return []

    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    merged = []

    for detection in sorted_detections:
        should_merge = False

        for i, existing in enumerate(merged):
            if calculate_iou(detection['bbox'], existing['bbox']) > iou_threshold:
                # Keep the one with higher confidence
                if detection['confidence'] > existing['confidence']:
                    merged[i] = detection
                should_merge = True
                break

        if not should_merge:
            merged.append(detection)

    return merged

def create_detection_summary(detections: List[dict]) -> str:
    """
    Create a text summary of detections.

    Args:
        detections (List[dict]): List of detections

    Returns:
        str: Summary text
    """
    if not detections:
        return "No vehicles detected."

    # Count by vehicle type
    vehicle_counts = {}
    total_confidence = 0

    for detection in detections:
        vehicle_type = detection['class_name']
        vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        total_confidence += detection['confidence']

    # Create summary
    summary_lines = []
    summary_lines.append(f"Total vehicles detected: {len(detections)}")

    for vehicle_type, count in vehicle_counts.items():
        summary_lines.append(f"  {vehicle_type.capitalize()}: {count}")

    avg_confidence = total_confidence / len(detections)
    summary_lines.append(f"Average confidence: {avg_confidence:.3f}")

    return "\n".join(summary_lines)

def validate_image_path(image_path: str) -> bool:
    """
    Validate if an image path exists and is a valid image file.

    Args:
        image_path (str): Path to the image

    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        return False

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(image_path).suffix.lower() in valid_extensions

def validate_video_path(video_path: str) -> bool:
    """
    Validate if a video path exists and is a valid video file.

    Args:
        video_path (str): Path to the video

    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False

    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return Path(video_path).suffix.lower() in valid_extensions

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a readable string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"
