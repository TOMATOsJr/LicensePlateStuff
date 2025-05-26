"""
Batch Vehicle Detection Script
Process multiple images or videos in batch mode.
"""

import os
import argparse
from pathlib import Path
import json
import time
from typing import List, Dict
import logging

from vehicle_detector import VehicleDetector
from utils import (
    ensure_directory, save_detections_json, validate_image_path,
    validate_video_path, format_time, create_detection_summary
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for vehicle detection on multiple files.
    """

    def __init__(self, confidence_threshold: float = 0.5, output_dir: str = "batch_output"):
        """
        Initialize batch processor.

        Args:
            confidence_threshold (float): Detection confidence threshold
            output_dir (str): Output directory for results
        """
        self.detector = VehicleDetector(confidence_threshold=confidence_threshold)
        self.output_dir = output_dir
        ensure_directory(output_dir)

        self.results = []
        self.start_time = None

    def process_images(self, image_paths: List[str], save_annotated: bool = True) -> Dict:
        """
        Process multiple images.

        Args:
            image_paths (List[str]): List of image file paths
            save_annotated (bool): Whether to save annotated images

        Returns:
            Dict: Processing results summary
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        self.start_time = time.time()

        results = {
            'total_files': len(image_paths),
            'processed_files': 0,
            'failed_files': 0,
            'total_vehicles': 0,
            'files': []
        }

        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")

            try:
                # Validate image
                if not validate_image_path(image_path):
                    logger.warning(f"Invalid image path: {image_path}")
                    results['failed_files'] += 1
                    continue

                # Detect vehicles
                output_path = None
                if save_annotated:
                    output_path = os.path.join(self.output_dir, f"detected_{Path(image_path).name}")

                detections = self.detector.detect_vehicles_image(
                    image_path,
                    save_result=save_annotated,
                    output_path=output_path
                )

                # Save detection results
                json_path = os.path.join(self.output_dir, f"{Path(image_path).stem}_detections.json")
                save_detections_json(detections, json_path)

                # Update results
                file_result = {
                    'file_path': image_path,
                    'file_name': Path(image_path).name,
                    'detections_count': len(detections),
                    'detections': detections,
                    'summary': create_detection_summary(detections)
                }

                results['files'].append(file_result)
                results['processed_files'] += 1
                results['total_vehicles'] += len(detections)

                logger.info(f"  Found {len(detections)} vehicles")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['failed_files'] += 1

        # Save batch results
        self._save_batch_results(results, "image_batch_results.json")

        elapsed_time = time.time() - self.start_time
        logger.info(f"Batch processing completed in {format_time(elapsed_time)}")

        return results

    def process_videos(self, video_paths: List[str], save_annotated: bool = True) -> Dict:
        """
        Process multiple videos.

        Args:
            video_paths (List[str]): List of video file paths
            save_annotated (bool): Whether to save annotated videos

        Returns:
            Dict: Processing results summary
        """
        logger.info(f"Starting batch processing of {len(video_paths)} videos")
        self.start_time = time.time()

        results = {
            'total_files': len(video_paths),
            'processed_files': 0,
            'failed_files': 0,
            'files': []
        }

        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {Path(video_path).name}")

            try:
                # Validate video
                if not validate_video_path(video_path):
                    logger.warning(f"Invalid video path: {video_path}")
                    results['failed_files'] += 1
                    continue

                # Process video
                output_path = None
                if save_annotated:
                    output_path = os.path.join(self.output_dir, f"detected_{Path(video_path).name}")

                self.detector.detect_vehicles_video(
                    video_path,
                    save_result=save_annotated,
                    output_path=output_path,
                    display_live=False
                )

                # Update results
                file_result = {
                    'file_path': video_path,
                    'file_name': Path(video_path).name,
                    'output_path': output_path,
                    'status': 'completed'
                }

                results['files'].append(file_result)
                results['processed_files'] += 1

                logger.info(f"  Video processing completed")

            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results['failed_files'] += 1

        # Save batch results
        self._save_batch_results(results, "video_batch_results.json")

        elapsed_time = time.time() - self.start_time
        logger.info(f"Batch processing completed in {format_time(elapsed_time)}")

        return results

    def _save_batch_results(self, results: Dict, filename: str):
        """Save batch processing results to file."""
        results_path = os.path.join(self.output_dir, filename)

        # Add timing information
        if self.start_time:
            results['processing_time'] = time.time() - self.start_time
            results['processing_time_formatted'] = format_time(results['processing_time'])

        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Error saving batch results: {e}")

def find_files_in_directory(directory: str, extensions: List[str]) -> List[str]:
    """
    Find all files with specified extensions in a directory.

    Args:
        directory (str): Directory to search
        extensions (List[str]): List of file extensions (e.g., ['.jpg', '.png'])

    Returns:
        List[str]: List of found file paths
    """
    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.error(f"Directory not found: {directory}")
        return files

    for ext in extensions:
        pattern = f"*{ext}"
        files.extend(directory_path.glob(pattern))
        files.extend(directory_path.glob(pattern.upper()))

    return [str(f) for f in sorted(files)]

def create_batch_report(results: Dict) -> str:
    """
    Create a text report from batch processing results.

    Args:
        results (Dict): Batch processing results

    Returns:
        str: Formatted report
    """
    report_lines = []
    report_lines.append("VEHICLE DETECTION BATCH PROCESSING REPORT")
    report_lines.append("=" * 50)

    # Summary
    report_lines.append(f"Total files: {results['total_files']}")
    report_lines.append(f"Successfully processed: {results['processed_files']}")
    report_lines.append(f"Failed: {results['failed_files']}")

    if 'total_vehicles' in results:
        report_lines.append(f"Total vehicles detected: {results['total_vehicles']}")

    if 'processing_time_formatted' in results:
        report_lines.append(f"Processing time: {results['processing_time_formatted']}")

    report_lines.append("")

    # File details
    if 'files' in results:
        report_lines.append("FILE DETAILS:")
        report_lines.append("-" * 20)

        for file_info in results['files']:
            report_lines.append(f"File: {file_info['file_name']}")

            if 'detections_count' in file_info:
                report_lines.append(f"  Vehicles detected: {file_info['detections_count']}")
                if 'summary' in file_info:
                    summary_lines = file_info['summary'].split('\n')
                    for line in summary_lines[1:]:  # Skip first line (total count)
                        report_lines.append(f"  {line}")
            elif 'status' in file_info:
                report_lines.append(f"  Status: {file_info['status']}")

            report_lines.append("")

    return "\n".join(report_lines)

def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description="Batch Vehicle Detection")
    parser.add_argument("--input", "-i", required=True,
                       help="Input directory or file path")
    parser.add_argument("--output", "-o", default="batch_output",
                       help="Output directory")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--type", "-t", choices=['images', 'videos', 'auto'],
                       default='auto', help="File type to process")
    parser.add_argument("--no-annotated", action="store_true",
                       help="Don't save annotated images/videos")
    parser.add_argument("--report", action="store_true",
                       help="Generate text report")

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(
        confidence_threshold=args.confidence,
        output_dir=args.output
    )

    # Determine file paths
    input_path = Path(args.input)

    if input_path.is_file():
        # Single file
        if args.type == 'images' or (args.type == 'auto' and
                                    validate_image_path(str(input_path))):
            file_paths = [str(input_path)]
            process_type = 'images'
        elif args.type == 'videos' or (args.type == 'auto' and
                                      validate_video_path(str(input_path))):
            file_paths = [str(input_path)]
            process_type = 'videos'
        else:
            logger.error("Invalid file type or unsupported file format")
            return

    elif input_path.is_dir():
        # Directory
        if args.type == 'images' or args.type == 'auto':
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = find_files_in_directory(str(input_path), image_extensions)

        if args.type == 'videos' or args.type == 'auto':
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = find_files_in_directory(str(input_path), video_extensions)

        if args.type == 'images':
            file_paths = image_files
            process_type = 'images'
        elif args.type == 'videos':
            file_paths = video_files
            process_type = 'videos'
        else:  # auto
            if image_files and video_files:
                print("Found both images and videos. Please specify --type")
                return
            elif image_files:
                file_paths = image_files
                process_type = 'images'
            elif video_files:
                file_paths = video_files
                process_type = 'videos'
            else:
                logger.error("No supported files found in directory")
                return
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return

    if not file_paths:
        logger.error("No files to process")
        return

    # Process files
    save_annotated = not args.no_annotated

    if process_type == 'images':
        results = processor.process_images(file_paths, save_annotated)
    else:
        results = processor.process_videos(file_paths, save_annotated)

    # Generate report
    if args.report:
        report = create_batch_report(results)
        report_path = os.path.join(args.output, "batch_report.txt")

        with open(report_path, 'w') as f:
            f.write(report)

        print(report)
        logger.info(f"Report saved to: {report_path}")

    # Print summary
    print(f"\nBatch processing completed!")
    print(f"Processed: {results['processed_files']}/{results['total_files']} files")
    if 'total_vehicles' in results:
        print(f"Total vehicles detected: {results['total_vehicles']}")

if __name__ == "__main__":
    main()
