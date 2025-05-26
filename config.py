"""
Configuration file for Vehicle Detection System
Modify these settings to customize the detection behavior.
"""

# Model Configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Can also use 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    'confidence_threshold': 0.5,  # Minimum confidence score (0.0 to 1.0)
    'iou_threshold': 0.45,        # IoU threshold for NMS
}

# Vehicle Classes (COCO dataset class IDs)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# Additional vehicle-related classes you might want to include
# 1: 'bicycle',  # Uncomment if you want to detect bicycles
# 4: 'airplane',  # Uncomment if you want to detect airplanes
# 6: 'train',     # Uncomment if you want to detect trains

# Display Configuration
DISPLAY_CONFIG = {
    'show_confidence': True,
    'show_class_name': True,
    'show_fps': True,
    'font_scale': 0.6,
    'font_thickness': 2,
    'box_thickness': 2,
}

# Color scheme for different vehicle types (BGR format)
COLORS = {
    'car': (0, 255, 0),        # Green
    'motorcycle': (255, 0, 0), # Blue
    'bus': (0, 0, 255),        # Red
    'truck': (255, 255, 0),    # Cyan
    'bicycle': (0, 255, 255),  # Yellow
    'default': (255, 255, 255) # White
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'output_fps': None,  # None = same as input, or specify fps
    'output_codec': 'mp4v',  # Video codec
    'show_progress': True,   # Show processing progress
    'progress_interval': 30, # Update progress every N frames
}

# Camera Configuration
CAMERA_CONFIG = {
    'default_camera_index': 0,
    'camera_resolution': None,  # None = default, or (width, height)
    'camera_fps': None,         # None = default
}

# File paths
PATHS = {
    'output_dir': 'output',     # Directory for output files
    'temp_dir': 'temp',         # Directory for temporary files
    'models_dir': 'models',     # Directory for model files
}

# Performance settings
PERFORMANCE = {
    'use_gpu': True,            # Use GPU if available
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
