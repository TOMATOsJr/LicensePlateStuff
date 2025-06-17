"""
Vehicle Detection System using YOLOv8
This module provides functionality to detect vehicles in images, videos, and live camera feeds.
It also detects license plates within detected vehicles.
"""

import cv2
import logging
from pathlib import Path
import time
from typing import List, Dict, Optional, Union

# Use absolute imports instead of relative imports
from models.vehicle_detector_model import VehicleDetectorModel
from models.license_plate_detector_model import LicensePlateDetectorModel
from models.ocr_model import OCRModel
from utils.image_utils import crop_vehicle, draw_detections, add_info_overlay
from utils.stats_utils import get_detection_statistics
from config import (
    DEFAULT_VEHICLE_MODEL_PATH,
    DEFAULT_LICENSE_PLATE_MODEL_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LICENSE_PLATE_CONFIDENCE,
    DEFAULT_MAX_VEHICLES,
    DEFAULT_DEVICE,
    OUTPUT_DIR
)

# Set up logging
logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    A vehicle detection system using YOLOv8n model.
    
    This class provides methods to detect vehicles in various input sources:
    - Static images
    - Video files
    - Live camera feeds
    
    It can also detect license plates within the detected vehicles.
    """
    
    def __init__(self,
                model_path: str = DEFAULT_VEHICLE_MODEL_PATH,
                confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                license_plate_model_path: str = DEFAULT_LICENSE_PLATE_MODEL_PATH,
                license_plate_confidence: float = DEFAULT_LICENSE_PLATE_CONFIDENCE,
                max_vehicles: int = DEFAULT_MAX_VEHICLES,
                device: str = DEFAULT_DEVICE):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path (str): Path to the YOLO model file
            confidence_threshold (float): Minimum confidence score for detections
            license_plate_model_path (str): Path to license plate detection model
            license_plate_confidence (float): Confidence threshold for license plate detection
            max_vehicles (int): Maximum number of vehicles to detect per frame
            device (str): Device to use for inference ('cpu' or 'cuda')
        """
        self.max_vehicles = max_vehicles
        
        # Initialize models
        self.vehicle_model = VehicleDetectorModel(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        self.license_plate_model = LicensePlateDetectorModel(
            model_path=license_plate_model_path,
            confidence_threshold=license_plate_confidence,
            device=device
        )
        
        self.ocr_model = OCRModel()
        
        logger.info("Vehicle detector initialized successfully")
    
    def detect_vehicles_image(self, image_path: str, save_result: bool = True,
                             output_path: str = None) -> List[dict]:
        """
        Detect vehicles in a single image.
        
        Args:
            image_path (str): Path to the input image
            save_result (bool): Whether to save the annotated image
            output_path (str): Path to save the result image
            
        Returns:
            List[dict]: List of detected vehicles with their information
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Run inference
            results = self.vehicle_model.predict(image)
            
            # Process results
            detections = self.vehicle_model.process_results(results[0], image.shape)
            
            # Limit to max_vehicles
            detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections
            
            # Detect license plates in each vehicle
            for detection in detections:
                vehicle_crop = crop_vehicle(image, detection['bbox'])
                license_plates = self.license_plate_model.detect_license_plates(vehicle_crop)
                
                # Perform OCR on each license plate
                for plate in license_plates:
                    plate_crop = crop_vehicle(vehicle_crop, plate['bbox'])
                    ocr_result = self.ocr_model.recognize_text(plate_crop)
                    plate['ocr_text'] = ocr_result['text']
                    plate['ocr_confidence'] = ocr_result['confidence']
                
                detection['license_plates'] = license_plates
            
            if save_result:
                # Create output directory if needed
                if output_path is None:
                    output_path = str(OUTPUT_DIR)
                Path(output_path).mkdir(exist_ok=True)
                
                annotated_image = draw_detections(image.copy(), detections)
                output_file = Path(output_path) / f"detected_{Path(image_path).name}"
                cv2.imwrite(str(output_file), annotated_image)
                logger.info(f"Annotated image saved to: {output_file}")
            
            logger.info(f"Detected {len(detections)} vehicles in image")
            return detections
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def detect_vehicles_video(self, video_path: str, save_result: bool = True,
                             output_path: str = None, display_live: bool = False) -> None:
        """
        Detect vehicles in a video file.
        
        Args:
            video_path (str): Path to the input video
            save_result (bool): Whether to save the annotated video
            output_path (str): Path to save the result video
            display_live (bool): Whether to display the video in real-time
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Set up video writer if saving
            out = None
            if save_result:
                # Create output directory if needed
                if output_path is None:
                    output_path = str(OUTPUT_DIR)
                Path(output_path).mkdir(exist_ok=True)
                
                output_file = Path(output_path) / f"detected_{Path(video_path).name}"
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                try:
                    start_inference = time.time()
                    results = self.vehicle_model.predict(frame)
                    detections = self.vehicle_model.process_results(results[0], frame.shape)
                    inference_time = time.time() - start_inference
                except Exception as inference_error:
                    logger.warning(f"Inference error on frame {frame_count}: {inference_error}")
                    # Skip this frame and continue
                    frame_count += 1
                    continue
                
                # Limit to max_vehicles
                detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections
                
                # Detect license plates in each vehicle
                for detection in detections:
                    vehicle_crop = crop_vehicle(frame, detection['bbox'])
                    license_plates = self.license_plate_model.detect_license_plates(vehicle_crop)
                    
                    # Perform OCR on each license plate
                    for plate in license_plates:
                        plate_crop = crop_vehicle(vehicle_crop, plate['bbox'])
                        ocr_result = self.ocr_model.recognize_text(plate_crop)
                        plate['ocr_text'] = ocr_result['text']
                        plate['ocr_confidence'] = ocr_result['confidence']
                    
                    detection['license_plates'] = license_plates
                
                # Draw detections
                annotated_frame = draw_detections(frame.copy(), detections)
                
                # Add frame info
                frame_info = {
                    'frame_count': frame_count,
                    'total_frames': total_frames,
                    'inference_time': inference_time,
                    'current_fps': frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
                }
                annotated_frame = add_info_overlay(annotated_frame, detections, frame_info)
                
                # Save frame
                if out:
                    out.write(annotated_frame)
                
                # Display frame
                if display_live:
                    cv2.imshow('Vehicle Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_count}/{total_frames} frames "
                              f"({fps_actual:.1f} FPS)")
            
            # Clean up
            cap.release()
            if out:
                out.release()
            if display_live:
                cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            logger.info(f"Video processing completed in {elapsed:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def detect_vehicles_camera(self, camera_index: int = 0, display_fps: bool = True, use_picamera: bool = False) -> None:
        """
        Detect vehicles from live camera feed.
        
        Args:
            camera_index (int): Camera index (usually 0 for default camera)
            display_fps (bool): Whether to display FPS information
            use_picamera (bool): Whether to use the Raspberry Pi camera module with libcamera
        """
        try:
            if use_picamera:
                # Use libcamera for Raspberry Pi - alternative approach using libcamera-still for snapshots
                try:
                    import subprocess
                    import numpy as np
                    import os
                    import tempfile
                    
                    logger.info("Starting Raspberry Pi camera detection with libcamera. Press 'q' to quit.")
                    
                    # FPS calculation variables
                    fps_counter = 0
                    fps_start_time = time.time()
                    current_fps = 0
                    
                    # Create a temporary file for image captures
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    running = True
                    while running:
                        # Capture a single frame using libcamera-jpeg
                        try:
                            subprocess.run([
                                "libcamera-jpeg", 
                                "-o", temp_path, 
                                "--width", "640", 
                                "--height", "480",
                                "-n",  # Don't show preview window
                                "-t", "1"  # Minimize timeout
                            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            # Read the saved image
                            frame = cv2.imread(temp_path)
                            if frame is None:
                                logger.warning("Failed to read frame from libcamera")
                                continue
                            
                            # Run inference
                            start_inference = time.time()
                            try:
                                results = self.vehicle_model.predict(frame)
                                detections = self.vehicle_model.process_results(results[0], frame.shape)
                                inference_time = time.time() - start_inference
                            except Exception as inference_error:
                                logger.warning(f"Inference error: {inference_error}")
                                detections = []
                                inference_time = time.time() - start_inference
                            
                            # Limit to max_vehicles
                            detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections
                            
                            # Detect license plates in each vehicle
                            for detection in detections:
                                vehicle_crop = crop_vehicle(frame, detection['bbox'])
                                license_plates = self.license_plate_model.detect_license_plates(vehicle_crop)
                                
                                # Perform OCR on each license plate
                                for plate in license_plates:
                                    plate_crop = crop_vehicle(vehicle_crop, plate['bbox'])
                                    ocr_result = self.ocr_model.recognize_text(plate_crop)
                                    plate['ocr_text'] = ocr_result['text']
                                    plate['ocr_confidence'] = ocr_result['confidence']
                                
                                detection['license_plates'] = license_plates
                            
                            # Draw detections
                            annotated_frame = draw_detections(frame.copy(), detections)
                            
                            # Calculate FPS
                            fps_counter += 1
                            if fps_counter >= 10:  # Update FPS every 10 frames
                                current_fps = fps_counter / (time.time() - fps_start_time) if time.time() - fps_start_time > 0 else 0
                                fps_counter = 0
                                fps_start_time = time.time()
                            
                            # Add information overlay
                            frame_info = {
                                'inference_time': inference_time,
                                'current_fps': current_fps
                            }
                            annotated_frame = add_info_overlay(annotated_frame, detections, frame_info, display_fps)
                            
                            # Display frame
                            cv2.imshow('Raspberry Pi Camera Detection', annotated_frame)
                            
                            # Check for quit command
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                running = False
                                
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Error calling libcamera: {e}")
                            time.sleep(1)  # Prevent tight loop if camera is having issues
                            
                except Exception as e:
                    logger.error(f"Error with libcamera: {e}")
                    raise
                
            else:
                # Original OpenCV camera code
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    raise ValueError(f"Could not open camera with index: {camera_index}")
                
                logger.info("Starting live vehicle detection. Press 'q' to quit.")
                
                # FPS calculation variables
                fps_counter = 0
                fps_start_time = time.time()
                current_fps = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame from camera")
                        continue
                    
                    # Run inference with error handling
                    start_inference = time.time()
                    try:
                        results = self.vehicle_model.predict(frame)
                        detections = self.vehicle_model.process_results(results[0], frame.shape)
                        inference_time = time.time() - start_inference
                    except Exception as inference_error:
                        logger.warning(f"Inference error: {inference_error}")
                        # Continue with empty detections
                        detections = []
                        inference_time = time.time() - start_inference
                    
                    # Limit to max_vehicles
                    detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections
                    
                    # Detect license plates in each vehicle
                    for detection in detections:
                        vehicle_crop = crop_vehicle(frame, detection['bbox'])
                        license_plates = self.license_plate_model.detect_license_plates(vehicle_crop)
                        
                        # Perform OCR on each license plate
                        for plate in license_plates:
                            plate_crop = crop_vehicle(vehicle_crop, plate['bbox'])
                            ocr_result = self.ocr_model.recognize_text(plate_crop)
                            plate['ocr_text'] = ocr_result['text']
                            plate['ocr_confidence'] = ocr_result['confidence']
                        
                        detection['license_plates'] = license_plates
                    
                    # Draw detections
                    annotated_frame = draw_detections(frame.copy(), detections)
                    
                    # Calculate FPS
                    fps_counter += 1
                    if fps_counter >= 30:
                        current_fps = fps_counter / (time.time() - fps_start_time) if time.time() - fps_start_time > 0 else 0
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    # Add information overlay
                    frame_info = {
                        'inference_time': inference_time,
                        'current_fps': current_fps
                    }
                    annotated_frame = add_info_overlay(annotated_frame, detections, frame_info, display_fps)
                    
                    # Display frame
                    cv2.imshow('Live Vehicle Detection', annotated_frame)
                    
                    # Check for quit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Clean up
                cap.release()
                cv2.destroyAllWindows()
                logger.info("Live detection stopped")
        
        except Exception as e:
            logger.error(f"Error in camera detection: {e}")
            raise
    
    def get_detection_statistics(self, detections: List[dict]) -> dict:
        """
        Get statistics about the detections.
        
        Args:
            detections: List of vehicle detections
            
        Returns:
            dict: Detection statistics
        """
        return get_detection_statistics(detections)
                