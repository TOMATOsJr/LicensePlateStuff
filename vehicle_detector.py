"""
Vehicle Detection System using YOLOv8n
This module provides functionality to detect vehicles in images, videos, and live camera feeds.
It also detects license plates within detected vehicles.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Union
import time
from paddleocr import PaddleOCR
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
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
                model_path: str = "yolov8n.pt",
                confidence_threshold: float = 0.5,
                license_plate_model_path: str = "licenseplatedetectyolo.pt",
                license_plate_confidence: float = 0.5,
                max_vehicles: int = 2,
                device: str = "cpu"):
        """
        Initialize the vehicle detector.

        Args:
            model_path (str): Path to the YOLO model file. Defaults to "yolov8n.pt"
            confidence_threshold (float): Minimum confidence score for detections
            license_plate_model_path (str): Path to license plate detection model
            license_plate_confidence (float): Confidence threshold for license plate detection
            max_vehicles (int): Maximum number of vehicles to detect per frame
            device (str): Device to use for inference ('cpu' or 'cuda'). Defaults to 'cpu'
        """
        self.model_path = model_path
        self.license_plate_model_path = license_plate_model_path
        self.confidence_threshold = confidence_threshold
        self.license_plate_confidence = license_plate_confidence
        self.max_vehicles = max_vehicles
        self.device = device
        self.model = None
        self.license_plate_model = None

        # Force CPU if CUDA has issues
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Vehicle class IDs in COCO dataset (used by YOLOv8)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 models."""
        try:
            logger.info(f"Loading vehicle detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Force model to use specified device
            self.model.to(self.device)
            logger.info(f"Vehicle model loaded successfully on {self.device}")

            logger.info(f"Loading license plate detection model: {self.license_plate_model_path}")
            self.license_plate_model = YOLO(self.license_plate_model_path)
            # Force model to use specified device
            self.license_plate_model.to(self.device)
            logger.info(f"License plate model loaded successfully on {self.device}")

            # Initialize PaddleOCR for license plate text recognition
            logger.info("Loading PaddleOCR model for text recognition")
            self.ocr = PaddleOCR(
                ocr_version="PP-OCRv3",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en"
            )
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If CUDA fails, try CPU
            if self.device == "cuda":
                logger.warning("CUDA model loading failed, retrying with CPU")
                self.device = "cpu"
                try:
                    self.model = YOLO(self.model_path)
                    self.model.to(self.device)
                    self.license_plate_model = YOLO(self.license_plate_model_path)
                    self.license_plate_model.to(self.device)

                    # Initialize PaddleOCR for license plate text recognition
                    self.ocr = PaddleOCR(
                        ocr_version="PP-OCRv4",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang="en"
                    )
                    logger.info("Models loaded successfully on CPU")
                except Exception as cpu_error:
                    logger.error(f"CPU model loading also failed: {cpu_error}")
                    raise
            else:
                raise

    def detect_vehicles_image(self, image_path: str, save_result: bool = True,
                             output_path: str = './output/') -> List[dict]:
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

            # Run inference with device specification
            results = self.model(image, conf=self.confidence_threshold, device=self.device)

            # Process results
            detections = self._process_results(results[0], image.shape)

            # Limit to max_vehicles
            detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections

            # Detect license plates in each vehicle
            for detection in detections:
                vehicle_crop = self._crop_vehicle(image, detection['bbox'])
                license_plates = self._detect_license_plate(vehicle_crop)
                detection['license_plates'] = license_plates

            if save_result:
                annotated_image = self._draw_detections(image.copy(), detections)
                if output_path:
                    output_file = f'{output_path}'+f'detected_{Path(image_path).name}'
                else:
                    output_file = f"detected_{Path(image_path).name}"
                cv2.imwrite(output_file, annotated_image)
                logger.info(f"Annotated image saved to: {output_file}")

            logger.info(f"Detected {len(detections)} vehicles in image")
            return detections

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def detect_vehicles_video(self, video_path: str, save_result: bool = True,
                             output_path: str = './output/', display_live: bool = False) -> None:
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
                if output_path:
                    output_file = f'{output_path}'+f'detected_{Path(video_path).name}'
                else:
                    output_file = f"detected_{Path(video_path).name}"
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference with device specification
                try:
                    results = self.model(frame, conf=self.confidence_threshold, device=self.device)
                    detections = self._process_results(results[0], frame.shape)
                except Exception as inference_error:
                    logger.warning(f"Inference error on frame {frame_count}: {inference_error}")
                    # Skip this frame and continue
                    frame_count += 1
                    continue

                # Limit to max_vehicles
                detections = detections[:self.max_vehicles] if len(detections) > self.max_vehicles else detections

                # Detect license plates in each vehicle
                for detection in detections:
                    vehicle_crop = self._crop_vehicle(frame, detection['bbox'])
                    license_plates = self._detect_license_plate(vehicle_crop)
                    detection['license_plates'] = license_plates

                # Draw detections
                annotated_frame = self._draw_detections(frame.copy(), detections)

                # Add frame info
                info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(detections)}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
                    fps_actual = frame_count / elapsed
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

    def detect_vehicles_camera(self, camera_index: int = 0, display_fps: bool = True) -> None:
        """
        Detect vehicles from live camera feed.

        Args:
            camera_index (int): Camera index (usually 0 for default camera)
            display_fps (bool): Whether to display FPS information
        """
        try:
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

                # Run inference with device specification and error handling
                start_inference = time.time()
                try:
                    results = self.model(frame, conf=self.confidence_threshold, device=self.device)
                    detections = self._process_results(results[0], frame.shape)
                except Exception as inference_error:
                    logger.warning(f"Inference error: {inference_error}")
                    # Continue with empty detections
                    detections = []                # Detect license plates in each vehicle
                total_plates = 0
                plates_with_text = 0
                detected_texts = []

                for detection in detections:
                    vehicle_crop = self._crop_vehicle(frame, detection['bbox'])
                    license_plates = self._detect_license_plate(vehicle_crop)
                    detection['license_plates'] = license_plates

                    # Count license plates and texts for overlay
                    total_plates += len(license_plates)
                    for plate in license_plates:
                        if 'ocr_text' in plate and plate['ocr_text'] and plate['ocr_text'].strip():
                            plates_with_text += 1
                            detected_texts.append(plate['ocr_text'])

                inference_time = time.time() - start_inference

                # Draw detections
                annotated_frame = self._draw_detections(frame.copy(), detections)

                # Add information overlay
                info_y = 30
                cv2.putText(annotated_frame, f"Vehicles detected: {len(detections)}",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                info_y += 30
                cv2.putText(annotated_frame, f"License plates: {total_plates} (with text: {plates_with_text})",
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Display detected license plate texts
                if detected_texts:
                    for i, text in enumerate(detected_texts[:3]):  # Show max 3 texts to avoid clutter
                        info_y += 30
                        cv2.putText(annotated_frame, f"Text {i+1}: {text}",
                                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if display_fps:
                    info_y += 30
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    info_y += 30
                    cv2.putText(annotated_frame, f"Inference: {inference_time*1000:.1f}ms",
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

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
            logger.error(f"Error in live detection: {e}")
            raise

    def _crop_vehicle(self, image: np.ndarray, bbox: dict) -> np.ndarray:
        """
        Crop a vehicle from the image based on bounding box.

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

    def _detect_license_plate(self, vehicle_image: np.ndarray) -> List[dict]:
        """
        Detect license plates in a vehicle image.

        Args:
            vehicle_image: Cropped vehicle image

        Returns:
            List[dict]: List of detected license plates
        """
        if vehicle_image.size == 0 or vehicle_image.shape[0] == 0 or vehicle_image.shape[1] == 0:
            return []        # Run inference on license plate model
        try:
            results = self.license_plate_model(vehicle_image, conf=self.license_plate_confidence, device=self.device)

            license_plates = []

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box

                    # Crop the license plate from the vehicle image
                    plate_crop = vehicle_image[int(y1):int(y2), int(x1):int(x2)]

                    # Perform OCR on the license plate crop
                    ocr_result = self._perform_ocr(plate_crop)

                    license_plate = {
                        'confidence': float(conf),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        },
                        'ocr_text': ocr_result['text'],
                        'ocr_confidence': ocr_result['confidence']
                    }
                    license_plates.append(license_plate)

            return license_plates
        except Exception as e:
            logger.warning(f"Error detecting license plate: {e}")
            return []

    def _perform_ocr(self, plate_image: np.ndarray) -> dict:
        """
        Perform OCR on a license plate image using PaddleOCR.

        Args:
            plate_image: Cropped license plate image

        Returns:
            dict: OCR result with text and confidence
        """
        try:
            # Check if the image is valid
            if plate_image.size == 0 or plate_image.shape[0] == 0 or plate_image.shape[1] == 0:
                return {'text': '', 'confidence': 0.0}

            # Convert numpy array to format expected by PaddleOCR
            # PaddleOCR expects BGR format (OpenCV default)
            if len(plate_image.shape) == 3 and plate_image.shape[2] == 3:
                # Image is already in BGR format
                ocr_input = plate_image
            else:
                # Convert grayscale to BGR if needed
                ocr_input = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR) if len(plate_image.shape) == 2 else plate_image            # Perform OCR using PaddleOCR
            results = self.ocr.predict(ocr_input)

            # Extract text and confidence from results
            if results and len(results) > 0:
                result = results[0]  # Get the first result

                # Access rec_texts and rec_scores from the result dictionary
                if isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
                    rec_texts = result['rec_texts']
                    rec_scores = result['rec_scores']

                    # Combine all valid text segments
                    valid_texts = []
                    total_confidence = 0.0

                    for text, confidence in zip(rec_texts, rec_scores):
                        if text and text.strip() and confidence > 0.5:  # Only use high-confidence text
                            valid_texts.append(text.strip())
                            total_confidence += confidence

                    if valid_texts:
                        # Join all valid text segments
                        recognized_text = ''.join(valid_texts)
                        avg_confidence = total_confidence / len(valid_texts)

                        logger.info(f"OCR Result: '{recognized_text}' (avg confidence: {avg_confidence:.3f})")

                        return {
                            'text': recognized_text,
                            'confidence': avg_confidence
                        }

            return {'text': '', 'confidence': 0.0}

        except Exception as e:
            logger.warning(f"Error performing OCR: {e}")
            return {'text': '', 'confidence': 0.0}

    def _process_results(self, results, image_shape: tuple) -> List[dict]:
        """
        Process YOLO results and filter for vehicles.

        Args:
            results: YOLO detection results
            image_shape: Shape of the input image (height, width, channels)

        Returns:
            List[dict]: Processed vehicle detections
        """
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box
                    detection = {
                        'class_id': int(class_id),
                        'class_name': self.vehicle_classes[class_id],
                        'confidence': float(conf),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        },
                        'license_plates': []
                    }
                    detections.append(detection)

            # Sort detections by confidence (highest first) and limit to max_vehicles
            detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def _draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on the image.

        Args:
            image: Input image
            detections: List of vehicle detections

        Returns:
            np.ndarray: Annotated image
        """
        # Color map for different vehicle types
        colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 0, 255),      # Red
            'truck': (255, 255, 0)   # Cyan
        }

        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']

            # Get color for this vehicle type
            color = colors.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(image,
                         (bbox['x1'], bbox['y1']),
                         (bbox['x2'], bbox['y2']),
                         color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Draw label background
            cv2.rectangle(image,
                         (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                         (bbox['x1'] + label_size[0], bbox['y1']),
                         color, -1)

            # Draw label text
            cv2.putText(image, label,
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
                    y2 = vehicle_y1 + plate_bbox['y2']                    # Draw license plate bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Draw license plate label with OCR text
                    if 'ocr_text' in plate and plate['ocr_text']:
                        plate_label = f"LP: {plate['ocr_text']} ({plate['confidence']:.2f})"
                    else:
                        plate_label = f"License Plate: {plate['confidence']:.2f}"

                    # Calculate label position (above the license plate box)
                    label_y = y1 - 5 if y1 > 20 else y2 + 20

                    cv2.putText(image, plate_label,
                               (x1, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return image

    def get_detection_statistics(self, detections: List[dict]) -> dict:
        """
        Get statistics about the detections.

        Args:
            detections: List of vehicle detections

        Returns:
            dict: Detection statistics
        """
        stats = {
            'total_vehicles': len(detections),
            'vehicle_types': {},
            'average_confidence': 0.0,
            'confidence_range': {'min': 1.0, 'max': 0.0},
            'license_plates_detected': 0,
            'license_plates_with_text': 0,
            'detected_texts': []
        }

        if detections:
            confidences = [d['confidence'] for d in detections]
            stats['average_confidence'] = np.mean(confidences)
            stats['confidence_range']['min'] = np.min(confidences)
            stats['confidence_range']['max'] = np.max(confidences)            # Count vehicle types and license plates
            for detection in detections:
                vehicle_type = detection['class_name']
                stats['vehicle_types'][vehicle_type] = stats['vehicle_types'].get(vehicle_type, 0) + 1
                if 'license_plates' in detection:
                    stats['license_plates_detected'] += len(detection['license_plates'])

                    # Count license plates with OCR text
                    for plate in detection['license_plates']:
                        if 'ocr_text' in plate and plate['ocr_text'] and plate['ocr_text'].strip():
                            stats['license_plates_with_text'] += 1
                            stats['detected_texts'].append({
                                'text': plate['ocr_text'],
                                'confidence': plate.get('ocr_confidence', 0.0)
                            })

        return stats


def main():
    """
    Main function demonstrating the vehicle detection system.
    """
    # Initialize detector with CPU device to avoid CUDA issues
    detector = VehicleDetector(confidence_threshold=0.5, device="cpu")

    print("Vehicle Detection System using YOLOv8n")
    print("=" * 40)
    print("1. Image detection")
    print("2. Video detection")
    print("3. Live camera detection")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                try:
                    detections = detector.detect_vehicles_image(image_path)
                    stats = detector.get_detection_statistics(detections)
                    print(f"\nDetection Results:")
                    print(f"Total vehicles: {stats['total_vehicles']}")
                    print(f"Vehicle types: {stats['vehicle_types']}")
                    print(f"Average confidence: {stats['average_confidence']:.3f}")
                    print(f"License plates detected: {stats['license_plates_detected']}")
                    print(f"License plates with text: {stats['license_plates_with_text']}")
                    if stats['detected_texts']:
                        print("Detected license plate texts:")
                        for i, text_info in enumerate(stats['detected_texts'], 1):
                            print(f"  {i}. '{text_info['text']}' (confidence: {text_info['confidence']:.3f})")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Image file not found!")

        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            if Path(video_path).exists():
                display = input("Display video while processing? (y/n): ").strip().lower() == 'y'
                try:
                    detector.detect_vehicles_video(video_path, display_live=display)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Video file not found!")

        elif choice == '3':
            camera_index = input("Enter camera index (default 0): ").strip()
            camera_index = int(camera_index) if camera_index.isdigit() else 0
            try:
                detector.detect_vehicles_camera(camera_index)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            print("Goodbye!")
            break

        else:
            print("Invalid choice! Please enter 1-4.")


if __name__ == "__main__":
    main()
