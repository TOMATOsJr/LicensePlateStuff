"""
Configuration module for the vehicle detection system.
Contains default settings and configuration parameters.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
OUTPUT_DIR = Path('./output/')
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configurations
DEFAULT_VEHICLE_MODEL_PATH = "yolov8n.pt"
DEFAULT_LICENSE_PLATE_MODEL_PATH = "licensePlatedetectyolo.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_LICENSE_PLATE_CONFIDENCE = 0.5
DEFAULT_MAX_VEHICLES = 2
DEFAULT_DEVICE = "cpu"

# Vehicle classes in COCO dataset (used by YOLOv8)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Color mapping for visualization
COLORS = {
    'car': (0, 255, 0),      # Green
    'motorcycle': (255, 0, 0),  # Blue
    'bus': (0, 0, 255),      # Red
    'truck': (255, 255, 0),   # Cyan
    'license_plate': (0, 255, 255)  # Yellow
}

# OCR Configuration
OCR_CONFIG = {
    'ocr_version': 'PP-OCRv3',
    'use_doc_orientation_classify': False,
    'use_doc_unwarping': False,
    'use_textline_orientation': False,
    'enable_mkldnn': False,
    'lang': 'en'
}

# OCR fallback configuration (when primary config fails)
OCR_FALLBACK_CONFIG = {
    'ocr_version': 'PP-OCRv4',
    'use_doc_orientation_classify': False,
    'use_doc_unwarping': False,
    'use_textline_orientation': False,
    'lang': 'en'
}

# Directory configurations
DIRECTORIES = {
    'output_dir': 'output',     # Directory for output files
    'temp_dir': 'temp',         # Directory for temporary files
    'models_dir': 'models',     # Directory for model files
}

# Performance settings
PERFORMANCE = {
    'use_gpu': False,            # Use GPU if available
    'batch_size': 1,            # Batch size for processing
    'num_workers': 0,           # Number of workers for data loading
}

# Logging configuration
LOGGING = {
    'level': 'INFO',            # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': False,         # Save logs to file
    'log_file': 'vehicle_detection.log'
}
