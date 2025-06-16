"""
Base detector class to provide common functionality for all detectors.
"""

import logging
import torch

logger = logging.getLogger(__name__)

class BaseDetector:
    """Base class for detector models."""
    
    def __init__(self, model_path: str, confidence_threshold: float, device: str):
        """
        Initialize base detector.
        
        Args:
            model_path (str): Path to model file
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._validate_device(device)
        self.model = None
    
    def _validate_device(self, device: str) -> str:
        """Validate and return appropriate device."""
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return device
    
    def _load_model(self):
        """Load model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model()")
    
    def predict(self, image):
        """
        Run inference on an image.
        
        Args:
            image: Input image
            
        Returns:
            Model prediction results
        """
        if self.model is None:
            self._load_model()
        
        try:
            results = self.model(image, conf=self.confidence_threshold, device=self.device)
            return results
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
