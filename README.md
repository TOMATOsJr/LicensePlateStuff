# Vehicle Detection System using YOLOv8n

This project implements a comprehensive vehicle detection system using YOLOv8n (You Only Look Once) for detecting vehicles in images, videos, and live camera feeds. The system is designed as a foundation for license plate recognition and traffic analysis applications.

## Features

- **Multi-source Detection**: Detect vehicles in static images, video files, and live camera feeds
- **Real-time Processing**: Optimized for real-time vehicle detection with FPS monitoring
- **Batch Processing**: Process multiple images or videos in batch mode
- **Vehicle Classification**: Detects and classifies cars, motorcycles, buses, and trucks
- **Configurable**: Easy configuration of detection parameters and display options
- **Export Results**: Save detection results as JSON and annotated images/videos
- **Performance Monitoring**: Built-in FPS calculation and processing time tracking

## Vehicle Classes Detected

The system can detect the following vehicle types using the COCO dataset classes:
- **Cars** (class ID: 2)
- **Motorcycles** (class ID: 3)
- **Buses** (class ID: 5)
- **Trucks** (class ID: 7)

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **The YOLOv8n model will be automatically downloaded on first run**

## Project Structure

```
LicensePlateStuff/
├── vehicle_detector.py      # Main vehicle detection class
├── batch_processor.py       # Batch processing script
├── examples.py             # Usage examples
├── utils.py                # Utility functions
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Quick Start

### Basic Usage

```python
from vehicle_detector import VehicleDetector

# Initialize detector
detector = VehicleDetector(confidence_threshold=0.5)

# Detect vehicles in an image
detections = detector.detect_vehicles_image("traffic.jpg")

# Detect vehicles in a video
detector.detect_vehicles_video("traffic.mp4", display_live=True)

# Start live camera detection
detector.detect_vehicles_camera()
```

### Command Line Usage

**Run the interactive detector:**
```bash
python vehicle_detector.py
```

**Batch process images:**
```bash
python batch_processor.py --input /path/to/images --output results --type images
```

**Batch process videos:**
```bash
python batch_processor.py --input /path/to/videos --output results --type videos
```

**Generate processing report:**
```bash
python batch_processor.py --input /path/to/files --report
```

## Detailed Usage

### 1. Image Detection

```python
from vehicle_detector import VehicleDetector

detector = VehicleDetector(confidence_threshold=0.5)

# Detect vehicles and save annotated image
detections = detector.detect_vehicles_image(
    image_path="traffic_scene.jpg",
    save_result=True,
    output_path="detected_traffic.jpg"
)

# Print detection results
for detection in detections:
    print(f"Vehicle: {detection['class_name']}")
    print(f"Confidence: {detection['confidence']:.3f}")
    print(f"Location: {detection['bbox']}")
```

### 2. Video Detection

```python
# Process video file
detector.detect_vehicles_video(
    video_path="traffic_video.mp4",
    save_result=True,
    output_path="detected_video.mp4",
    display_live=True  # Show processing in real-time
)
```

### 3. Live Camera Detection

```python
# Start live detection from default camera
detector.detect_vehicles_camera(
    camera_index=0,      # Use camera 0
    display_fps=True     # Show FPS information
)
```

### 4. Batch Processing

```python
from batch_processor import BatchProcessor

# Initialize batch processor
processor = BatchProcessor(
    confidence_threshold=0.5,
    output_dir="batch_results"
)

# Process multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = processor.process_images(image_paths, save_annotated=True)

# Process multiple videos
video_paths = ["vid1.mp4", "vid2.mp4"]
results = processor.process_videos(video_paths, save_annotated=True)
```

## Configuration

You can customize the detection behavior by modifying `config.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    'model_path': 'yolov8n.pt',  # Model size: n, s, m, l, x
    'confidence_threshold': 0.5,  # Detection confidence
}

# Display Configuration
DISPLAY_CONFIG = {
    'show_confidence': True,
    'show_class_name': True,
    'font_scale': 0.6,
    'box_thickness': 2,
}

# Colors for different vehicle types
COLORS = {
    'car': (0, 255, 0),        # Green
    'motorcycle': (255, 0, 0), # Blue
    'bus': (0, 0, 255),        # Red
    'truck': (255, 255, 0),    # Cyan
}
```

## Advanced Features

### Detection Statistics

```python
# Get detailed statistics about detections
stats = detector.get_detection_statistics(detections)
print(f"Total vehicles: {stats['total_vehicles']}")
print(f"Vehicle types: {stats['vehicle_types']}")
print(f"Average confidence: {stats['average_confidence']:.3f}")
```

### Filtering and Processing

```python
from utils import filter_detections_by_size, merge_overlapping_detections

# Filter small detections
filtered = filter_detections_by_size(detections, min_width=50, min_height=50)

# Merge overlapping detections
merged = merge_overlapping_detections(detections, iou_threshold=0.5)

# Crop vehicle regions
from utils import crop_vehicle_region
for detection in detections:
    vehicle_crop = crop_vehicle_region(image, detection['bbox'], padding=10)
```

### Saving Results

```python
from utils import save_detections_json, create_detection_summary

# Save detections as JSON
save_detections_json(detections, "results.json")

# Create text summary
summary = create_detection_summary(detections)
print(summary)
```

## Performance Tips

1. **Model Selection**: Use different YOLO models for speed vs accuracy trade-off:
   - `yolov8n.pt`: Fastest, smallest
   - `yolov8s.pt`: Small
   - `yolov8m.pt`: Medium
   - `yolov8l.pt`: Large
   - `yolov8x.pt`: Extra Large, most accurate

2. **GPU Acceleration**: The system automatically uses GPU if available (CUDA)

3. **Confidence Threshold**: Lower values detect more objects but may include false positives

4. **Batch Processing**: Use batch processing for multiple files to improve efficiency

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'ultralytics'**
   ```bash
   pip install ultralytics
   ```

2. **Camera not working**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure camera is not being used by other applications

3. **Low FPS during processing**
   - Use a smaller YOLO model (yolov8n.pt)
   - Reduce input resolution
   - Enable GPU acceleration

4. **Out of memory errors**
   - Use yolov8n.pt model
   - Process smaller batches
   - Reduce input image/video resolution

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **Storage**: 2GB for models and dependencies

## Integration with License Plate Recognition

This vehicle detection system is designed to work as the first stage in a license plate recognition pipeline:

1. **Vehicle Detection**: Identify vehicle regions in the image
2. **License Plate Detection**: Search for license plates within detected vehicle regions
3. **OCR**: Extract text from detected license plates

The detected vehicle bounding boxes can be used to:
- Focus license plate detection on relevant areas
- Improve accuracy by filtering out non-vehicle regions
- Associate license plates with specific vehicles

## Examples and Demos

Run the examples script to see demonstrations:

```bash
python examples.py
```

This will create sample scripts and demonstrate various usage patterns.

## Contributing

Feel free to contribute to this project by:
- Reporting bugs and issues
- Suggesting new features
- Submitting pull requests
- Improving documentation

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation
- [OpenCV](https://opencv.org/) for computer vision functionality
- The COCO dataset for providing the training data used by YOLO models