"""
Main script for the Vehicle Detection System.
This provides a command-line interface to the system.
"""

import logging
from pathlib import Path
from detector import VehicleDetector
from config import DEFAULT_CONFIDENCE_THRESHOLD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function demonstrating the vehicle detection system.
    """
    # Initialize detector with CPU device to avoid CUDA issues
    detector = VehicleDetector(confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, device="cpu")
    
    print("\nVehicle Detection System using YOLOv8n")
    print("=" * 40)
    print("1. Image detection")
    print("2. Video detection")
    print("3. Live camera detection (Raspberry Pi)")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if Path(image_path).exists():
                try:
                    detections = detector.detect_vehicles_image(image_path)
                    stats = detector.get_detection_statistics(detections)
                    print(f"\nDetection Results:")
                    print(f"Total vehicles: {stats['total_vehicles']}")
                    print(f"Vehicle types: {stats['vehicle_types']}")
                    print(f"Average confidence: {stats['average_confidence']:.3f}")
                    print(f"License plates detected: {stats['license_plates_detected']}")
                    print(f"License plates with text: {stats['license_plates_with_text']}")
                    if stats['detected_texts']:
                        print("Detected license plate texts:")
                        for i, text_info in enumerate(stats['detected_texts'], 1):
                            print(f"  {i}. '{text_info['text']}' (confidence: {text_info['confidence']:.3f})")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Image file not found!")
        
        elif choice == '2':
            video_path = input("Enter video path: ").strip()
            if Path(video_path).exists():
                display = input("Display video while processing? (y/n): ").strip().lower() == 'y'
                try:
                    detector.detect_vehicles_video(video_path, display_live=display)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Video file not found!")
        
        elif choice == '3':
            try:
                print("Starting Raspberry Pi camera detection using libcamera...")
                detector.detect_vehicles_camera(use_picamera=True)
            except Exception as e:
                print(f"Error with camera: {e}")
                print("If libcamera is not installed, install it with:")
                print("sudo apt-get update && sudo apt-get install -y libcamera-apps")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-4.")

if __name__ == "__main__":
    main()
