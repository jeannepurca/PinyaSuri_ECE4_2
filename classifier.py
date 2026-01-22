#!/usr/bin/env python3
# classifier.py - OBJECT DETECTION VERSION (YOLOv8 TFLite)

import logging
import numpy as np
import cv2
import json
import config

logger = logging.getLogger(__name__)

class PinyaSuriAI:    
    def __init__(self):
        try:
            import tflite_runtime.interpreter as tflite
            
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
            
            # Load the model
            self.interpreter = tflite.Interpreter(model_path=str(config.MODEL_PATH))
            self.interpreter.allocate_tensors()
            
            # Get input/output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get expected input shape
            self.input_shape = self.input_details[0]['shape']
            self.input_height = self.input_shape[1]
            self.input_width = self.input_shape[2]
            
            # Try to load class names from model metadata
            self.class_names = self._load_class_names_from_model()
            
            if self.class_names:
                logger.info(f"✓ Loaded {len(self.class_names)} class names from model metadata")
                for idx, name in self.class_names.items():
                    logger.info(f"  Class {idx}: {name}")
            else:
                logger.warning("⚠ No class names in model metadata, using defaults from config")
                self.class_names = config.CLASS_NAMES
            
            logger.info(f"✓ Object Detection Model loaded: {config.MODEL_PATH.name}")
            logger.info(f"  Input shape: {self.input_shape}")
            logger.info(f"  Number of classes: {len(self.class_names)}")
            logger.info(f"  Detection threshold: {config.DETECTION_THRESHOLD}")
            
        except ImportError:
            logger.error("⚠ tflite_runtime not installed!")
            logger.error("  Install with: pip3 install tflite-runtime")
            raise
            
        except Exception as e:
            logger.error(f"⚠ Failed to load detection model: {e}")
            raise

    def _load_class_names_from_model(self):
        """
        Try to extract class names from TFLite model metadata
        
        Returns:
            dict: Class index to name mapping, or None if not available
        """
        try:
            model_path = config.MODEL_PATH
            
            # Try to read associated metadata files
            metadata_json = model_path.parent / f"{model_path.stem}_metadata.json"
            labels_txt = model_path.parent / f"{model_path.stem}_labels.txt"
            
            # Try JSON metadata file
            if metadata_json.exists():
                with open(metadata_json, 'r') as f:
                    metadata = json.load(f)
                    if 'class_names' in metadata:
                        return {i: name for i, name in enumerate(metadata['class_names'])}
            
            # Try labels.txt file
            if labels_txt.exists():
                with open(labels_txt, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]
                    return {i: name for i, name in enumerate(labels)}
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not load class names from model metadata: {e}")
            return None

    def get_class_name(self, index: int) -> str:
        """Get class name for a given index"""
        return config.get_class_name(index, self.class_names)

    def preprocess_frame(self, frame):
        """Preprocess frame for YOLOv8 input"""
        try:
            # YOLOv8 expects RGB (OpenCV loads as BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
            
            # Normalize to [0, 1] and convert to float32
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension: [1, height, width, 3]
            input_data = np.expand_dims(normalized, axis=0)
            
            return input_data
            
        except Exception as e:
            logger.error(f"⚠ Preprocessing failed: {e}")
            return None

    def detect(self, frame):
        """
        Detect multiple pineapples in a frame (YOLOv8 format)
        
        Returns: List of detections
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Preprocess
            input_data = self.preprocess_frame(frame)
            if input_data is None:
                return []
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # YOLOv8 has 1 output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # YOLOv8 output shape: [1, num_features, num_boxes]
            # For 7 classes: [1, 11, 8400] where 11 = [x, y, w, h, class1...class7]
            
            logger.debug(f"YOLOv8 output shape: {output_data.shape}")
            
            # Transpose to [num_boxes, num_features]
            predictions = output_data[0].T  # Shape: [8400, 11]
            
            detections = []
            
            for pred in predictions:
                # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
                x_center, y_center, width, height = pred[:4]
                
                # Get class scores
                class_scores = pred[4:]
                class_idx = int(np.argmax(class_scores))
                confidence = float(class_scores[class_idx])
                
                # Filter by confidence threshold
                if confidence < config.DETECTION_THRESHOLD:
                    continue
                
                # Get class name
                class_name = self.get_class_name(class_idx)
                
                # YOLOv8 coordinates are already normalized [0, 1] relative to input size
                # Convert from center format to corner format
                xmin = (x_center - width / 2) / self.input_width
                ymin = (y_center - height / 2) / self.input_height
                xmax = (x_center + width / 2) / self.input_width
                ymax = (y_center + height / 2) / self.input_height
                
                # Clamp to [0, 1]
                xmin = max(0.0, min(1.0, xmin))
                ymin = max(0.0, min(1.0, ymin))
                xmax = max(0.0, min(1.0, xmax))
                ymax = max(0.0, min(1.0, ymax))
                
                # Convert to pixel coordinates
                x1_px = int(xmin * frame_width)
                y1_px = int(ymin * frame_height)
                x2_px = int(xmax * frame_width)
                y2_px = int(ymax * frame_height)
                
                detection = {
                    'class_index': class_idx,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (xmin, ymin, xmax, ymax),  # normalized
                    'bbox_pixels': (x1_px, y1_px, x2_px, y2_px)  # pixels
                }
                
                detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} pineapple(s)")
            
            return detections
            
        except Exception as e:
            logger.error(f"⚠ Detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def detect_with_nms(self, frame, iou_threshold=0.5):
        """
        Detect with Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            frame: Input image
            iou_threshold: IoU threshold for NMS (default 0.5)
        
        Returns: Filtered list of detections
        """
        detections = self.detect(frame)
        
        if len(detections) <= 1:
            return detections
        
        # Apply NMS
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        indices = self._nms(boxes, scores, iou_threshold)
        
        filtered_detections = [detections[i] for i in indices]
        
        logger.debug(f"NMS: {len(detections)} → {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def _nms(self, boxes, scores, iou_threshold):
        """Simple Non-Maximum Suppression implementation"""
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        
        while len(sorted_indices) > 0:
            # Pick box with highest score
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # Keep boxes with IoU below threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        # box: [xmin, ymin, xmax, ymax]
        # boxes: [N, 4]
        
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = box_area + boxes_area - intersection
        
        iou = intersection / (union + 1e-6)
        
        return iou

    def get_detection_summary(self, detections):
        """Get summary statistics from detections"""
        if not detections:
            return {
                'total_count': 0,
                'class_counts': {},
                'avg_confidence': 0.0
            }
        
        class_counts = {}
        total_confidence = 0.0
        
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += det['confidence']
        
        return {
            'total_count': len(detections),
            'class_counts': class_counts,
            'avg_confidence': total_confidence / len(detections)
        }