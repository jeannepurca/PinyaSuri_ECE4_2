#!/usr/bin/env python3
# classifier.py - OBJECT DETECTION VERSION

import logging
import numpy as np
import cv2
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
            
            logger.info(f"✓ Object Detection Model loaded: {config.MODEL_PATH.name}")
            logger.info(f"  Input shape: {self.input_shape}")
            logger.info(f"  Number of classes: {len(config.CLASS_NAMES)}")
            logger.info(f"  Detection threshold: {config.DETECTION_THRESHOLD}")
            
        except ImportError:
            logger.error("⚠ tflite_runtime not installed!")
            logger.error("  Install with: pip3 install tflite-runtime")
            raise
            
        except Exception as e:
            logger.error(f"⚠ Failed to load detection model: {e}")
            raise

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        try:
            # Resize to model input size
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(normalized, axis=0)
            
            return input_data
            
        except Exception as e:
            logger.error(f"⚠ Preprocessing failed: {e}")
            return None

    def detect(self, frame):
        """
        Detect multiple pineapples in a frame
        
        Returns: List of detections, each containing:
            {
                'class_index': int,
                'class_name': str,
                'confidence': float,
                'bbox': (x1, y1, x2, y2),  # normalized [0-1]
                'bbox_pixels': (x1, y1, x2, y2)  # actual pixels
            }
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
            
            # Get outputs
            # For TFLite object detection models, typical outputs are:
            # output[0]: bounding boxes (normalized coordinates)
            # output[1]: class indices
            # output[2]: confidence scores
            # output[3]: number of detections
            
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # [N, 4]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # [N]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # [N]
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])
            
            detections = []
            
            for i in range(num_detections):
                score = float(scores[i])
                
                # Filter by confidence threshold
                if score < config.DETECTION_THRESHOLD:
                    continue
                
                class_idx = int(classes[i])
                class_name = config.get_class_name(class_idx)
                
                # Bounding box (normalized coordinates: ymin, xmin, ymax, xmax)
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Convert to pixel coordinates
                x1_px = int(xmin * frame_width)
                y1_px = int(ymin * frame_height)
                x2_px = int(xmax * frame_width)
                y2_px = int(ymax * frame_height)
                
                detection = {
                    'class_index': class_idx,
                    'class_name': class_name,
                    'confidence': score,
                    'bbox': (xmin, ymin, xmax, ymax),  # normalized
                    'bbox_pixels': (x1_px, y1_px, x2_px, y2_px)  # pixels
                }
                
                detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} pineapple(s)")
            
            return detections
            
        except Exception as e:
            logger.error(f"⚠ Detection failed: {e}")
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
        
        # Convert to [x1, y1, x2, y2] format for NMS
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