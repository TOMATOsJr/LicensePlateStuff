"""
Vehicle detection model using YOLOv8.
"""

import logging
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

from models.base_detector import BaseDetector
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)

class VehicleDetectorModel(BaseDetector):
    """Vehicle detection model using YOLOv8."""
    
    def __init__(self, model_path: str, confidence_threshold: float, device: str):
        """
        Initialize vehicle detector model.
        
        Args:
            model_path (str): Path to YOLOv8 model file
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__(model_path, confidence_threshold, device)
        self.vehicle_classes = VEHICLE_CLASSES
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            logger.info(f"Loading vehicle detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Force model to use specified device
            self.model.to(self.device)
            logger.info(f"Vehicle model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading vehicle model: {e}")
            if self.device == "cuda":
                logger.warning("CUDA model loading failed, retrying with CPU")
                self.device = "cpu"
                try:
                    self.model = YOLO(self.model_path)
                    self.model.to(self.device)
                    logger.info("Vehicle model loaded successfully on CPU")
                except Exception as cpu_error:
                    logger.error(f"CPU model loading also failed: {cpu_error}")
                    raise
            else:
                raise
    
    def process_results(self, results, image_shape: tuple) -> List[dict]:
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
            
            # Sort detections by confidence (highest first)
            detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
