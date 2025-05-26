"""
Example usage of the Vehicle Detection System
This script demonstrates how to use the VehicleDetector class for different scenarios.
"""

from vehicle_detector import VehicleDetector
import cv2
import numpy as np
from pathlib import Path

def example_image_detection():
    """Example of detecting vehicles in a static image."""
    print("=== Image Detection Example ===")

    # Initialize detector
    detector = VehicleDetector(confidence_threshold=0.4)

    # Example: Create a sample image path (you should replace with actual image)
    image_path = "sample_traffic.jpg"

    if Path(image_path).exists():
        # Detect vehicles
        detections = detector.detect_vehicles_image(image_path, save_result=True)

        # Print results
        print(f"Found {len(detections)} vehicles:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.3f})")
            bbox = detection['bbox']
            print(f"      Location: ({bbox['x1']}, {bbox['y1']}) to ({bbox['x2']}, {bbox['y2']})")

        # Get statistics
        stats = detector.get_detection_statistics(detections)
        print(f"\nStatistics:")
        print(f"  Total vehicles: {stats['total_vehicles']}")
        print(f"  Vehicle types: {stats['vehicle_types']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")
    else:
        print(f"Image file '{image_path}' not found. Please provide a valid image path.")

def example_video_detection():
    """Example of detecting vehicles in a video file."""
    print("\n=== Video Detection Example ===")

    # Initialize detector
    detector = VehicleDetector(confidence_threshold=0.5)

    # Example video path (you should replace with actual video)
    video_path = "traffic_video.mp4"

    if Path(video_path).exists():
        print(f"Processing video: {video_path}")
        print("This will create an annotated output video...")

        # Process video (set display_live=True to see real-time processing)
        detector.detect_vehicles_video(
            video_path,
            save_result=True,
            output_path="detected_traffic_video.mp4",
            display_live=False  # Set to True if you want to see live processing
        )
        print("Video processing completed!")
    else:
        print(f"Video file '{video_path}' not found. Please provide a valid video path.")

def example_camera_detection():
    """Example of live vehicle detection from camera."""
    print("\n=== Live Camera Detection Example ===")

    # Initialize detector
    detector = VehicleDetector(confidence_threshold=0.5)

    print("Starting live camera detection...")
    print("Press 'q' in the video window to quit")

    try:
        # Start live detection (camera index 0 is usually the default camera)
        detector.detect_vehicles_camera(camera_index=0, display_fps=True)
    except Exception as e:
        print(f"Error with camera detection: {e}")
        print("Make sure you have a camera connected and accessible.")

def create_sample_detection_script():
    """Create a simple detection script for quick testing."""
    print("\n=== Creating Sample Detection Script ===")

    # This would be useful for testing with your own images/videos
    script_content = '''
# Quick Vehicle Detection Script
from vehicle_detector import VehicleDetector

# Initialize detector
detector = VehicleDetector(confidence_threshold=0.5)

# Detect in image
# detections = detector.detect_vehicles_image("your_image.jpg")

# Detect in video
# detector.detect_vehicles_video("your_video.mp4", display_live=True)

# Live camera detection
# detector.detect_vehicles_camera()

print("Vehicle Detection System Ready!")
print("Uncomment the lines above and provide your own image/video paths")
'''

    with open("quick_detect.py", "w") as f:
        f.write(script_content)

    print("Created 'quick_detect.py' for easy testing")

def test_with_sample_image():
    """Create a simple test image and detect vehicles (for demonstration)."""
    print("\n=== Creating Test Image ===")

    # Create a simple test image with some shapes (not real vehicles, just for testing)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some rectangles to simulate objects
    cv2.rectangle(test_image, (100, 200), (200, 300), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_image, (300, 250), (450, 350), (255, 0, 0), -1)  # Blue rectangle
    cv2.putText(test_image, "Test Image - Use real traffic images for actual detection",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save test image
    cv2.imwrite("test_image.jpg", test_image)
    print("Created 'test_image.jpg' for testing")
    print("Note: This is just a test image. Use real traffic images for actual vehicle detection.")

def main():
    """Main function to run all examples."""
    print("Vehicle Detection System - Examples")
    print("=" * 50)

    # Create test resources
    create_sample_detection_script()
    test_with_sample_image()

    print("\nTo test the vehicle detection system:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Run: python vehicle_detector.py")
    print("3. Or use the created 'quick_detect.py' script")
    print("4. For best results, use real traffic images or videos")

    print("\nExample usage in code:")
    print("=" * 30)

    # Show code examples
    print("""
# Basic usage
from vehicle_detector import VehicleDetector

# Initialize
detector = VehicleDetector(confidence_threshold=0.5)

# Detect in image
detections = detector.detect_vehicles_image("traffic.jpg")

# Detect in video
detector.detect_vehicles_video("traffic.mp4", display_live=True)

# Live camera detection
detector.detect_vehicles_camera()
""")

if __name__ == "__main__":
    main()
