"""
License plate detection model using YOLOv8.
"""

import logging
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

from models.base_detector import BaseDetector

logger = logging.getLogger(__name__)

class LicensePlateDetectorModel(BaseDetector):
    """License plate detection model using YOLOv8."""
    
    def __init__(self, model_path: str, confidence_threshold: float, device: str):
        """
        Initialize license plate detector model.
        
        Args:
            model_path (str): Path to license plate model file
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__(model_path, confidence_threshold, device)
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 license plate model."""
        try:
            logger.info(f"Loading license plate detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            # Force model to use specified device
            self.model.to(self.device)
            logger.info(f"License plate model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading license plate model: {e}")
            if self.device == "cuda":
                logger.warning("CUDA model loading failed, retrying with CPU")
                self.device = "cpu"
                try:
                    self.model = YOLO(self.model_path)
                    self.model.to(self.device)
                    logger.info("License plate model loaded successfully on CPU")
                except Exception as cpu_error:
                    logger.error(f"CPU model loading also failed: {cpu_error}")
                    raise
            else:
                raise
    
    def detect_license_plates(self, vehicle_image: np.ndarray) -> List[dict]:
        """
        Detect license plates in a vehicle image.
        
        Args:
            vehicle_image: Cropped vehicle image
            
        Returns:
            List[dict]: List of detected license plates
        """
        if vehicle_image.size == 0 or vehicle_image.shape[0] == 0 or vehicle_image.shape[1] == 0:
            return []
        
        try:
            results = self.predict(vehicle_image)
            license_plates = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    
                    license_plate = {
                        'confidence': float(conf),
                        'bbox': {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                    }
                    license_plates.append(license_plate)
            
            return license_plates
        except Exception as e:
            logger.warning(f"Error detecting license plate: {e}")
            return []
