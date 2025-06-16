"""
Utility functions for generating statistics from detections.
"""

import numpy as np
from typing import List, Dict

def get_detection_statistics(detections: List[dict]) -> dict:
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
        stats['confidence_range']['max'] = np.max(confidences)
        
        # Count vehicle types and license plates
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
