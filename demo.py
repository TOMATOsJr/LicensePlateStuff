"""
Quick Demo of Vehicle Detection System
This script demonstrates the vehicle detection capabilities.
"""

import cv2
import numpy as np
from vehicle_detector import VehicleDetector
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_image():
    """Create a simple test image for demonstration."""
    # Create a test image with some basic shapes
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some background
    img[:] = (50, 50, 50)  # Dark gray background

    # Add some rectangles to simulate vehicles
    cv2.rectangle(img, (100, 200), (250, 300), (100, 150, 200), -1)  # Car-like shape
    cv2.rectangle(img, (300, 180), (500, 320), (150, 100, 200), -1)  # Larger vehicle
    cv2.rectangle(img, (400, 350), (550, 420), (200, 100, 100), -1)  # Another vehicle

    # Add some text
    cv2.putText(img, "Test Image for Vehicle Detection Demo",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Note: Use real traffic images for actual detection",
                (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img

def demo_image_detection():
    """Demonstrate vehicle detection on a test image."""
    print("🚗 Vehicle Detection Demo - Image Processing")
    print("=" * 50)

    # Create test image
    test_img = create_test_image()
    test_path = "demo_test_image.jpg"
    cv2.imwrite(test_path, test_img)
    print(f"✓ Created test image: {test_path}")

    # Initialize detector
    print("📡 Initializing YOLOv8n detector...")
    detector = VehicleDetector(confidence_threshold=0.3)
    print("✓ Detector initialized successfully")

    # Detect vehicles
    print("🔍 Running vehicle detection...")
    detections = detector.detect_vehicles_image(
        test_path,
        save_result=True,
        output_path="demo_detected_image.jpg"
    )

    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Total detections: {len(detections)}")

    if detections:
        print("   Detected vehicles:")
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            print(f"     {i+1}. {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.3f}) "
                  f"at ({bbox['x1']}, {bbox['y1']}) - ({bbox['x2']}, {bbox['y2']})")
    else:
        print("   No vehicles detected in test image")
        print("   💡 This is expected as the test image contains simple shapes")
        print("   💡 Try with real traffic images for actual vehicle detection")

    # Get statistics
    stats = detector.get_detection_statistics(detections)
    print(f"\n📈 Statistics:")
    print(f"   Vehicle types: {stats['vehicle_types']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")

    print(f"\n✅ Demo completed!")
    print(f"   Check 'demo_detected_image.jpg' for annotated results")

def demo_camera_info():
    """Show information about camera detection."""
    print("\n📹 Camera Detection Information")
    print("=" * 35)
    print("To test live camera detection, run:")
    print("   python vehicle_detector.py")
    print("Then select option 3 for live camera detection")
    print("\nCamera requirements:")
    print("   • Connected camera (webcam, USB camera, etc.)")
    print("   • Camera permissions granted")
    print("   • No other applications using the camera")
    print("\nControls during live detection:")
    print("   • Press 'q' to quit")
    print("   • Detection runs in real-time with FPS display")

def demo_batch_processing():
    """Show information about batch processing."""
    print("\n📁 Batch Processing Information")
    print("=" * 35)
    print("For processing multiple files:")
    print("   python batch_processor.py --input /path/to/images --type images")
    print("   python batch_processor.py --input /path/to/videos --type videos")
    print("\nBatch processing features:")
    print("   • Process entire directories")
    print("   • Generate detailed reports")
    print("   • Save results in JSON format")
    print("   • Annotated output files")

def main():
    """Main demo function."""
    print("🚗 VEHICLE DETECTION SYSTEM DEMO")
    print("=" * 50)
    print("This demo shows the capabilities of the YOLOv8n-based")
    print("vehicle detection system for license plate recognition.")
    print()

    try:
        # Demo image detection
        demo_image_detection()

        # Show other capabilities
        demo_camera_info()
        demo_batch_processing()

        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("=" * 50)

        print("\nNext steps:")
        print("1. Test with real traffic images/videos")
        print("2. Try live camera detection")
        print("3. Use batch processing for multiple files")
        print("4. Integrate with license plate recognition")

        print("\nUseful commands:")
        print("   python vehicle_detector.py          # Interactive mode")
        print("   python examples.py                  # View examples")
        print("   python batch_processor.py --help    # Batch processing help")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure all dependencies are installed correctly")

if __name__ == "__main__":
    main()
