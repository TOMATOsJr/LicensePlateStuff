"""
Setup script for Vehicle Detection System
This script helps set up the environment and download necessary models.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required!")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "output",
        "temp",
        "models",
        "batch_output"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def test_installation():
    """Test if the installation is working."""
    logger.info("Testing installation...")
    try:
        # Try importing required modules
        import cv2
        import numpy as np
        from ultralytics import YOLO

        logger.info("✓ OpenCV imported successfully")
        logger.info("✓ NumPy imported successfully")
        logger.info("✓ Ultralytics imported successfully")

        # Try loading YOLO model (this will download it if not present)
        logger.info("Downloading YOLOv8n model (this may take a few minutes)...")
        model = YOLO("yolov8n.pt")
        logger.info("✓ YOLOv8n model loaded successfully")

        # Test basic detection capability
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model(test_image)
        logger.info("✓ Detection test passed")

        return True

    except Exception as e:
        logger.error(f"Installation test failed: {e}")
        return False

def check_gpu_support():
    """Check if GPU support is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU support available: {gpu_name} ({gpu_count} device(s))")
        else:
            logger.info("GPU support not available - will use CPU")
    except ImportError:
        logger.warning("PyTorch not available - cannot check GPU support")

def main():
    """Main setup function."""
    print("Vehicle Detection System Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Create directories
    create_directories()

    # Check GPU support
    check_gpu_support()

    # Test installation
    if not test_installation():
        logger.error("Setup completed with errors. Please check the logs above.")
        sys.exit(1)

    logger.info("\n" + "=" * 50)
    logger.info("🎉 Setup completed successfully!")
    logger.info("=" * 50)

    print("\nNext steps:")
    print("1. Run 'python vehicle_detector.py' for interactive detection")
    print("2. Run 'python examples.py' to see usage examples")
    print("3. Place your images/videos in the project folder and start detecting!")

    print("\nQuick test:")
    print("python -c \"from vehicle_detector import VehicleDetector; print('System ready!')\"")

if __name__ == "__main__":
    main()
