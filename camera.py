#!/usr/bin/env python3
# camera.py - OBJECT DETECTION VERSION

import logging
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self):
        config.ensure_directories()

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            
            # Configure for fast continuous capture
            cam_config = self.picam2.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                buffer_count=3
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            
            logger.info("✓ Camera started in streaming mode!")
            logger.info(f"  Resolution: 640x480")
            
        except Exception as e:
            logger.error(f"⚠ Failed to initialize camera: {e} ⚠")
            raise

    def capture_array(self):
        """Capture a frame as numpy array for immediate inference"""
        try:
            frame = self.picam2.capture_array()
            
            # Ensure RGB format (remove alpha if present)
            if frame.shape[-1] == 4:  # RGBA
                frame = frame[:, :, :3]  # Keep only RGB
            
            return frame
            
        except Exception as e:
            logger.error(f"⚠ Failed to capture frame: {e}")
            return None

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: RGB numpy array
            detections: List of detection dictionaries
        
        Returns:
            Annotated frame (RGB)
        """
        # Convert RGB to BGR for OpenCV
        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox_pixels']
            class_name = det['class_name']
            confidence = det['confidence']
            class_idx = det['class_index']
            
            # Get color for this class
            color = config.get_class_color(class_idx)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
            
            # Prepare label text
            label = f"{class_name}: {confidence*100:.1f}%"
            
            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA
            )
        
        # Convert back to RGB
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        return annotated_rgb

    def save_detection_image(self, frame, detections, waypoint, flight_number, burst_index=0):
        """
        Save a frame with detections drawn on it
        
        Args:
            frame: Original RGB frame
            detections: List of detection dictionaries
            waypoint: Waypoint number
            flight_number: Flight number
            burst_index: Frame index in burst
        
        Returns:
            Path to saved image
        """
        date_folder = config.get_image_day_dir()
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
        
        # Count detections per class
        det_summary = self._get_detection_filename_summary(detections)
        
        # Filename with detection summary
        filename = (f"pinyasuri_flight{flight_number}_wp{waypoint}_"
                   f"burst{burst_index}_{det_summary}_{ts}.jpg")
        fullpath = date_folder / filename

        try:
            # Draw bounding boxes if enabled
            if config.DRAW_BBOXES and detections:
                annotated_frame = self.draw_detections(frame, detections)
            else:
                annotated_frame = frame
            
            # Convert to PIL and save
            img = Image.fromarray(annotated_frame)
            img.save(str(fullpath), quality=95)
            
            logger.debug(f"✓ Saved detection image: {filename}")
            return str(fullpath)
            
        except Exception as e:
            logger.error(f"⚠ Failed to save detection image: {e}")
            return None

    def _get_detection_filename_summary(self, detections):
        """
        Create concise summary for filename
        Example: "2H_1MW_1CR" = 2 Healthy, 1 Mealybug Wilt, 1 Crown Rot
        """
        if not detections:
            return "0det"
        
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Abbreviations
        abbrev = {
            "Healthy": "H",
            "Mealybug Wilt Disease": "MW",
            "Root Rot Disease": "RR",
            "Crown Rot Disease": "CR",
            "Fruit Fasciation Disorder": "FF",
            "Multiple Crown Disorder": "MC"
        }
        
        summary_parts = []
        for class_name, count in sorted(class_counts.items()):
            short = abbrev.get(class_name, "UK")
            summary_parts.append(f"{count}{short}")
        
        return "_".join(summary_parts) if summary_parts else "0det"

    def close(self):
        try:
            self.picam2.stop()
            logger.info("✓ Camera stopped successfully.")
        except Exception as e:
            logger.warning(f"⚠ Error stopping camera: {e} ⚠")