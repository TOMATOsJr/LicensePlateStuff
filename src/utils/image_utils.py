"""
Utility functions for image processing.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple

from config import COLORS

logger = logging.getLogger(__name__)

def crop_vehicle(image: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Crop a vehicle from an image based on bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box dictionary with x1, y1, x2, y2
    
    Returns:
        np.ndarray: Cropped vehicle image
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Ensure crop coordinates are within image boundaries
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    return image[y1:y2, x1:x2]

def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draw detection boxes and labels on the image.
    
    Args:
        image: Input image
        detections: List of vehicle detections
    
    Returns:
        np.ndarray: Annotated image
    """
    # Make a copy of the image
    annotated_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Get color for this vehicle type
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_image,
                     (bbox['x1'], bbox['y1']),
                     (bbox['x2'], bbox['y2']),
                     color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(annotated_image,
                     (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                     (bbox['x1'] + label_size[0], bbox['y1']),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated_image, label,
                   (bbox['x1'], bbox['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw license plate detections (in vehicle coordinates)
        if 'license_plates' in detection and detection['license_plates']:
            vehicle_x1, vehicle_y1 = bbox['x1'], bbox['y1']
            
            for i, plate in enumerate(detection['license_plates']):
                plate_bbox = plate['bbox']
                
                # Convert plate coordinates to image coordinates
                x1 = vehicle_x1 + plate_bbox['x1']
                y1 = vehicle_y1 + plate_bbox['y1']
                x2 = vehicle_x1 + plate_bbox['x2']
                y2 = vehicle_y1 + plate_bbox['y2']
                
                # Draw license plate bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), COLORS['license_plate'], 2)
                
                # Draw license plate label with OCR text
                if 'ocr_text' in plate and plate['ocr_text']:
                    plate_label = f"LP: {plate['ocr_text']} ({plate['confidence']:.2f})"
                else:
                    plate_label = f"License Plate: {plate['confidence']:.2f}"
                
                # Calculate label position (above the license plate box)
                label_y = y1 - 5 if y1 > 20 else y2 + 20
                
                cv2.putText(annotated_image, plate_label,
                           (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['license_plate'], 2)
    
    return annotated_image

def add_info_overlay(frame: np.ndarray, 
                     detections: List[dict], 
                     frame_info: dict = None, 
                     display_fps: bool = True) -> np.ndarray:
    """
    Add information overlay to a video frame.
    
    Args:
        frame: The input frame
        detections: List of vehicle detections
        frame_info: Additional frame information (optional)
        display_fps: Whether to display FPS information
    
    Returns:
        np.ndarray: Frame with overlay information
    """
    # Count license plates and texts for overlay
    total_plates = 0
    plates_with_text = 0
    detected_texts = []
    
    for detection in detections:
        if 'license_plates' in detection:
            total_plates += len(detection['license_plates'])
            for plate in detection['license_plates']:
                if 'ocr_text' in plate and plate['ocr_text'] and plate['ocr_text'].strip():
                    plates_with_text += 1
                    detected_texts.append(plate['ocr_text'])
    
    # Add information overlay
    info_y = 30
    cv2.putText(frame, f"Vehicles detected: {len(detections)}",
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    info_y += 30
    cv2.putText(frame, f"License plates: {total_plates} (with text: {plates_with_text})",
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display detected license plate texts
    if detected_texts:
        for i, text in enumerate(detected_texts[:3]):  # Show max 3 texts to avoid clutter
            info_y += 30
            cv2.putText(frame, f"Text {i+1}: {text}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Add frame info if provided
    if frame_info:
        if 'current_fps' in frame_info and display_fps:
            info_y += 30
            cv2.putText(frame, f"FPS: {frame_info['current_fps']:.1f}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 'inference_time' in frame_info:
            info_y += 30
            cv2.putText(frame, f"Inference: {frame_info['inference_time']*1000:.1f}ms",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if 'frame_count' in frame_info and 'total_frames' in frame_info:
            info_y += 30
            cv2.putText(frame, f"Frame: {frame_info['frame_count']}/{frame_info['total_frames']}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame
