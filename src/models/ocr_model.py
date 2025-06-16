"""
OCR model for license plate text recognition.
"""

import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict

from config import OCR_CONFIG, OCR_FALLBACK_CONFIG

logger = logging.getLogger(__name__)

class OCRModel:
    """OCR model for license plate text recognition."""
    
    def __init__(self):
        """Initialize OCR model."""
        self.ocr = None
        self._load_model()
    
    def _load_model(self):
        """Load the OCR model."""
        try:
            logger.info("Loading PaddleOCR model for text recognition")
            self.ocr = PaddleOCR(**OCR_CONFIG)
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PaddleOCR model: {e}")
            try:
                logger.info("Trying fallback OCR configuration")
                self.ocr = PaddleOCR(**OCR_FALLBACK_CONFIG)
                logger.info("PaddleOCR fallback model loaded successfully")
            except Exception as fallback_error:
                logger.error(f"Fallback OCR model loading also failed: {fallback_error}")
                raise
    
    def recognize_text(self, plate_image: np.ndarray) -> Dict:
        """
        Perform OCR on a license plate image.
        
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
                ocr_input = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR) if len(plate_image.shape) == 2 else plate_image
            
            # Perform OCR using PaddleOCR
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
