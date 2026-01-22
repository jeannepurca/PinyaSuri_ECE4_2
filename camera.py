#!/usr/bin/env python3
# camera.py

import logging
import pathlib
from datetime import datetime
import config
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self):
        config.ensure_directories()

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            
            # Use faster configuration optimized for burst capture
            # Still uses high resolution but with faster processing
            cam_config = self.picam2.create_still_configuration(
                main={"size": (4056, 3040)},
                buffer_count=2  # Double buffering for faster captures
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            
            logger.info("✓ Camera started successfully!")
            
        except Exception as e:
            logger.error(f"⚠ Failed to initialize camera: {e} ⚠")
            raise

    def capture(self, waypoint: int, flight_number: int = 1, prefix="img", burst_index=0):
        """Capture image and save to today's date folder"""

        # Get today's folder
        date_folder = config.get_image_day_dir()

        # Timestamp for filename
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
        
        # Include burst index in filename
        filename = f"{prefix}_flight{flight_number}_wp{waypoint}_burst{burst_index}_{ts}.jpg"
        fullpath = date_folder / filename

        try:
            self.picam2.capture_file(str(fullpath))
            logger.debug(f"✓ Captured {filename}")
            return str(fullpath)
            
        except Exception as e:
            logger.error(f"⚠ Failed to capture {filename}: {e}")
            raise

    def capture_array(self):
        """Capture frame as numpy array for streaming inference"""
        try:
            # Capture as RGB array
            frame = self.picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
            
        except Exception as e:
            logger.error(f"⚠ Failed to capture frame array: {e}")
            return None

    def save_detection_image(self, frame, detections, waypoint: int, flight_number: int = 1, 
                            prefix="detection", burst_index=0):
        """
        Save image with bounding boxes drawn on detections
        
        Args:
            frame: OpenCV BGR image array
            detections: List of detection dictionaries from classifier
            waypoint: Waypoint number
            flight_number: Flight number
            prefix: Filename prefix
            burst_index: Burst image index
            
        Returns:
            str: Path to saved image, or None if failed
        """
        try:
            # Get today's folder
            date_folder = config.get_image_day_dir()

            # Timestamp for filename
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
            
            # Create filename
            filename = f"{prefix}_flight{flight_number}_wp{waypoint}_burst{burst_index}_{ts}.jpg"
            fullpath = date_folder / filename

            # Draw bounding boxes if enabled
            if config.DRAW_BBOXES and detections:
                frame = self._draw_bounding_boxes(frame.copy(), detections)
            
            # Save image
            cv2.imwrite(str(fullpath), frame)
            logger.debug(f"✓ Saved detection image: {filename}")
            return str(fullpath)
            
        except Exception as e:
            logger.error(f"⚠ Failed to save detection image: {e}")
            return None

    def _draw_bounding_boxes(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: OpenCV BGR image
            detections: List of detection dictionaries
            
        Returns:
            Frame with bounding boxes drawn
        """
        frame_height, frame_width = frame.shape[:2]
        
        for det in detections:
            # Get bounding box in pixels
            x1, y1, x2, y2 = det['bbox_pixels']
            
            # Get class info
            class_idx = det['class_index']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for this class
            color = config.get_class_color(class_idx)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                config.FONT_SCALE, 
                1
            )
            
            # Draw label background (filled rectangle)
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(
                frame,
                (x1, label_y - text_height - baseline),
                (x1 + text_width, label_y + baseline),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        return frame

    def close(self):
        try:
            self.picam2.stop()
            logger.info("✓ Camera stopped successfully.")
        except Exception as e:
            logger.warning(f"⚠ Error stopping camera: {e} ⚠")