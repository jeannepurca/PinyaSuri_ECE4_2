#!/usr/bin/env python3
# main_stream.py - STREAMING INFERENCE VERSION

import time
import csv
import logging
import sys
from pathlib import Path
import json
import config
import uploader
from logging_config import setup_logging
from pixhawk import Pixhawk
from camera import Camera
from classifier import PinyaSuriAI
from metrics import get_next_daily_flight_number
from metrics import FlightMetricsLogger

running = True

# ----------------------------
# CSV Initialization
# ----------------------------
def initialize_image_log():
    """Create image log CSV with headers if file doesn't exist"""
    if not config.IMAGE_LOG_CSV.exists():
        with open(config.IMAGE_LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "flight_id",
                "flight_number",
                "waypoint",
                "lat_deg",
                "lon_deg",
                "rel_alt_m",
                "burst_id",
                "burst_index",
                "num_detections",
                "image_path"
            ])

def initialize_detection_csv():
    """Create detection CSV with headers if not exists"""
    if not config.CLASSIFICATION_CSV.exists():
        with open(config.CLASSIFICATION_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "flight_id",
                "flight_number",
                "waypoint",
                "lat_deg",
                "lon_deg",
                "rel_alt_m",
                "burst_id",
                "burst_index",
                "detection_index",
                "class_index",
                "class_name",
                "confidence",
                "bbox_xmin",
                "bbox_ymin",
                "bbox_xmax",
                "bbox_ymax",
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
                "image_path"
            ])

def log_detections(flight_id, flight_number, waypoint, position, burst_id, burst_index, 
                   detections, image_path, logger):
    """Log all detections from a single frame to CSV"""
    try:
        with open(config.CLASSIFICATION_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            
            for det_idx, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det['bbox']
                x1_px, y1_px, x2_px, y2_px = det['bbox_pixels']
                
                writer.writerow([
                    time.time(),
                    flight_id,
                    flight_number,
                    waypoint,
                    position["lat"],
                    position["lon"],
                    position["rel_alt"],
                    burst_id,
                    burst_index,
                    det_idx,
                    det['class_index'],
                    det['class_name'],
                    f"{det['confidence']:.4f}",
                    f"{xmin:.6f}",
                    f"{ymin:.6f}",
                    f"{xmax:.6f}",
                    f"{ymax:.6f}",
                    x1_px,
                    y1_px,
                    x2_px,
                    y2_px,
                    image_path
                ])
        
        # Also add to uploader for JSON summary
        uploader.add_detection_to_flight(flight_id, waypoint, image_path, detections)
        
    except Exception as e:
        logger.error(f"Failed to log detections: {e}")

# ----------------------------
# Streaming Inference Handler
# ----------------------------
def handle_waypoint_streaming_detection(pixhawk, camera, classifier, metrics, waypoint, 
                                        flight_number, captured_wp, logger):
    """Stream frames and detect objects in real-time at waypoint"""
    if waypoint in captured_wp:
        return False
        
    wp_name = config.get_waypoint_name(waypoint)
    logger.info("=" * 60)
    logger.info(f">>> {wp_name} (WP{waypoint}) REACHED - Starting object detection...")
    logger.info("=" * 60)
    
    # Wait for drone to fully stabilize
    logger.info(f"⏳ Waiting {config.STABILIZATION_DELAY}s for drone to settle...")
    stabilization_start = time.time()
    
    while (time.time() - stabilization_start) < config.STABILIZATION_DELAY:
        pixhawk.update()
        
        if pixhawk.mode != "AUTO":
            logger.warning("=" * 60)
            logger.warning(f"⚠️ DETECTION ABORTED - Mode changed to {pixhawk.mode}")
            logger.warning("=" * 60)
            return False
        
        time.sleep(0.1)

    pixhawk.update()
    
    if pixhawk.mode != "AUTO":
        logger.warning("=" * 60)
        logger.warning(f"⚠️ DETECTION ABORTED - Not in AUTO mode ({pixhawk.mode})")
        logger.warning("=" * 60)
        return False
    
    # Stream and detect frames
    num_frames = config.BURST_CAPTURE_COUNT
    frame_interval = config.BURST_INTERVAL
    detection_results = []
    burst_id = f"{metrics.flight_id}_wp{waypoint}_{int(time.time())}"

    logger.info(f"🤖 Starting object detection ({num_frames} frames)...")

    for i in range(num_frames):
        # Check mode before each frame
        pixhawk.update()
        if pixhawk.mode != "AUTO":
            logger.warning("=" * 60)
            logger.warning(f"⚠️ DETECTION ABORTED at frame {i+1}/{num_frames} - Mode changed to {pixhawk.mode}")
            logger.warning(f"   Detected objects in {len(detection_results)} frames before abort")
            logger.warning("=" * 60)
            
            if detection_results:
                captured_wp.add(waypoint)
            return len(detection_results) > 0
        
        try:
            # Capture frame
            frame = camera.capture_array()
            if frame is None:
                logger.error(f"⚠ Frame {i+1} capture failed")
                continue
            
            # Detect objects with NMS
            detections = classifier.detect_with_nms(frame, iou_threshold=config.NMS_IOU_THRESHOLD)
            
            # Save image with bounding boxes
            image_path = camera.save_detection_image(
                frame, detections, waypoint, flight_number, burst_index=i
            )
            
            if image_path:
                detection_results.append({
                    "frame_index": i,
                    "num_detections": len(detections),
                    "detections": detections,
                    "image_path": image_path
                })
                
                # Log each detection to CSV and uploader
                log_detections(
                    metrics.flight_id,
                    flight_number,
                    waypoint,
                    pixhawk.position,
                    burst_id,
                    i,
                    detections,
                    image_path,
                    logger
                )
                
                # Display frame summary
                if detections:
                    det_summary = classifier.get_detection_summary(detections)
                    logger.info(f"  ✓ Frame {i+1}/{num_frames}: {det_summary['total_count']} objects detected")
                    for class_name, count in det_summary['class_counts'].items():
                        logger.info(f"      - {class_name}: {count}")
                else:
                    logger.info(f"  ✓ Frame {i+1}/{num_frames}: No objects detected")
            else:
                logger.error(f"⚠ Frame {i+1} failed to save")
            
            if i < num_frames - 1:
                time.sleep(frame_interval)
                
        except Exception as e:
            logger.error(f"⚠ Frame {i+1} detection failed: {e}")

    if detection_results:
        # Calculate overall statistics
        total_detections = sum(r['num_detections'] for r in detection_results)
        frames_with_detections = sum(1 for r in detection_results if r['num_detections'] > 0)
        
        logger.info(f"✓ PROCESSED {len(detection_results)}/{num_frames} frames at {wp_name}")
        logger.info(f"  Burst ID: {burst_id}")
        logger.info(f"  Total objects detected: {total_detections}")
        logger.info(f"  Frames with detections: {frames_with_detections}/{len(detection_results)}")
        
        # Aggregate class counts across all frames
        all_class_counts = {}
        for result in detection_results:
            for det in result['detections']:
                class_name = det['class_name']
                all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
        
        if all_class_counts:
            logger.info("  Overall Detection Summary:")
            for class_name, count in sorted(all_class_counts.items()):
                logger.info(f"    {class_name}: {count} detection(s)")
        
        captured_wp.add(waypoint)
        return True
    else:
        logger.error("⚠ Object detection completely failed - no valid results")
        return False

def get_telemetry_dict(pixhawk):
    """Build telemetry dictionary for metrics"""
    if not pixhawk.position:
        return None
    
    return {
        "rel_alt": pixhawk.position["rel_alt"],
        "lat": pixhawk.position["lat"],
        "lon": pixhawk.position["lon"],
        "imu_accel": pixhawk.imu_accel,
    }

def handle_arm_state_change(pixhawk, metrics, was_armed, flight_number, captured_wp, logger):
    """Detect and handle arm/disarm transitions"""
    if pixhawk.armed and not was_armed:
        # Just armed
        logger.info("=" * 60)
        logger.info(f"🛫 FLIGHT #{flight_number} - DRONE ARMED")
        logger.info("   Mission monitoring started.")
        logger.info("=" * 60)
        metrics.start_flight()
        pixhawk.clear_waypoint_log()
        
        return True, flight_number
        
    elif not pixhawk.armed and was_armed:
        # Just disarmed
        logger.info(f"🛬 FLIGHT #{flight_number} - DRONE DISARMED")
        metrics.end_flight()
        captured_wp.clear()
        pixhawk.clear_waypoint_log()
        
        return False, metrics.flight_number
    
    return was_armed, flight_number

def is_drone_in_air(pixhawk):
    """Check if drone altitude is within capture range"""
    if not pixhawk.position:
        return False
    
    alt = pixhawk.position["rel_alt"]
    
    # Check if altitude is within the acceptable range
    in_range = config.MIN_ALTITUDE_FOR_CAPTURE <= alt <= config.MAX_ALTITUDE_FOR_CAPTURE
    
    return in_range

def should_capture_image(pixhawk, waypoint, captured_wp, logger):
    """Check if drone should perform inference at current waypoint"""

    # 1. Must be armed
    if not pixhawk.armed:
        logger.debug("❌ Check failed: Not armed")
        return False
    
    # 2. Must be in AUTO mode
    if pixhawk.mode != "AUTO":
        logger.debug(f"❌ Check failed: Not in AUTO mode (current: {pixhawk.mode})")
        return False
    
    # 3. Must have valid waypoint
    if not waypoint:
        logger.debug("❌ Check failed: No waypoint")
        return False
    
    # 4. Must have position data from GPS
    if not pixhawk.position:
        logger.debug("❌ Check failed: No position data")
        return False
    
    # 5. Must be in the air
    if not is_drone_in_air(pixhawk):
        alt = pixhawk.position['rel_alt'] if pixhawk.position else 0
        logger.debug(f"❌ Check failed: Altitude {alt:.2f}m not in range "
                    f"[{config.MIN_ALTITUDE_FOR_CAPTURE}m - {config.MAX_ALTITUDE_FOR_CAPTURE}m]")
        return False
    
    # 6. Must NOT have already captured this waypoint
    if waypoint in captured_wp:
        logger.debug(f"❌ Check failed: Already captured WP{waypoint}")
        return False
    
    # 7. Must capture only at survey/mapping waypoints (exclude last waypoint)
    last_wp = pixhawk.get_last_waypoint()
    if not config.is_mapping_waypoint(waypoint, last_wp):
        logger.debug(f"❌ Check failed: WP{waypoint} is not a mapping waypoint")
        return False

    # 8. Must have distance data
    if pixhawk.wp_dist is None:
        logger.debug(f"❌ Check failed: No distance data available yet")
        return False
    
    # 9. Must be hovering
    if pixhawk.groundspeed > config.HOVER_SPEED_THRESHOLD:
        logger.debug(f"❌ Check failed: Still moving at {pixhawk.groundspeed:.2f} m/s "
                    f"(threshold: {config.HOVER_SPEED_THRESHOLD} m/s)")
        return False
    
    # 10. Must be within capture distance
    if pixhawk.wp_dist > config.WAYPOINT_CAPTURE_DISTANCE:
        logger.debug(f"❌ Check failed: Too far from WP{waypoint} "
                    f"({pixhawk.wp_dist:.2f}m > {config.WAYPOINT_CAPTURE_DISTANCE}m)")
        return False

    # All checks passed!
    logger.info("=" * 60)
    logger.info(f"✅ ALL CHECKS PASSED - Triggering inference for WP{waypoint}!")
    logger.info(f"   Altitude: {pixhawk.position['rel_alt']:.2f}m")
    logger.info(f"   Distance to waypoint: {pixhawk.wp_dist:.2f}m")
    logger.info(f"   Groundspeed: {pixhawk.groundspeed:.2f} m/s")
    logger.info("=" * 60)
    return True


def main_loop(pixhawk, camera, classifier, metrics, logger):
    captured_wp = set()
    flight_number = metrics.flight_number
    was_armed = False
    current_mode = "UNKNOWN"
    current_waypoint = None
    last_debug_time = 0
    
    logger.info("=" * 60)
    logger.info("🍍 PINYASURI FLIGHT SYSTEM READY! 🚁")
    logger.info("System will run continuously. Press Ctrl+C to stop.")
    logger.info("=" * 60)

    while running:
        # Update pixhawk telemetry
        pixhawk.update()
        
        # Handle arm/disarm state changes
        was_armed, flight_number = handle_arm_state_change(
            pixhawk, metrics, was_armed, flight_number, captured_wp, logger
        )

        # Check for flight mode changes
        if pixhawk.mode != current_mode:
            current_mode = pixhawk.mode
            logger.info(f"> Flight Mode: {current_mode}")

        # Check for waypoint changes (only when armed and waypoint is valid)
        if pixhawk.armed and pixhawk.last_wp is not None and pixhawk.last_wp != current_waypoint:
            current_waypoint = pixhawk.last_wp
            wp_name = config.get_waypoint_name(current_waypoint)
            wp_type = config.get_waypoint_type(current_waypoint)
            
            logger.info(f"📍 Navigating to {wp_name} (WP{current_waypoint}) [{wp_type}]")

        # PERIODIC DEBUG OUTPUT (every 2 seconds when armed)
        if pixhawk.armed and (time.time() - last_debug_time) > 2.0:
            last_debug_time = time.time()
            dist_str = f"{pixhawk.wp_dist:.2f}m" if pixhawk.wp_dist else "N/A"
            
            # ✅ FIX: Check if position exists before accessing
            if pixhawk.position:
                alt_str = f"{pixhawk.position['rel_alt']:.1f}m"
            else:
                alt_str = "N/A"
            
            logger.debug(f"[STATUS] Mode: {pixhawk.mode}, WP: {pixhawk.last_wp}, "
                        f"Alt: {alt_str}, "
                        f"Dist to WP: {dist_str}, "
                        f"Captured: {captured_wp}")

        # Safe Waypoint Guard
        wp = pixhawk.current_waypoint or {"lat": None, "lon": None, "alt": None}

        # Update metrics during flight
        if pixhawk.armed and pixhawk.position:  # ✅ FIX: Added position check
            telemetry = {
                "attitude": {
                    "roll": getattr(pixhawk, "roll", 0.0),
                    "pitch": getattr(pixhawk, "pitch", 0.0),
                    "yaw": getattr(pixhawk, "yaw", 0.0)
                },
                "imu_accel": pixhawk.imu_accel,
                "gps": {
                    "lat": pixhawk.position["lat"],
                    "lon": pixhawk.position["lon"],
                    "alt": pixhawk.position["rel_alt"],
                    "groundspeed": pixhawk.groundspeed
                },
                "waypoint_index": pixhawk.last_wp,
                "waypoint_lat": wp["lat"],
                "waypoint_lon": wp["lon"],
                "waypoint_alt": wp["alt"],
                "flight_mode": pixhawk.mode,
                "nav_state": pixhawk.nav_state,
                "is_hovering": pixhawk.is_hovering(threshold=config.HOVER_SPEED_THRESHOLD),
            }
            metrics.log_telemetry(telemetry)

        
        # Check for streaming inference
        if should_capture_image(pixhawk, pixhawk.last_wp, captured_wp, logger):
            handle_waypoint_streaming_detection(
                pixhawk, camera, classifier, metrics, 
                pixhawk.last_wp, flight_number, captured_wp, logger
            )
        
        time.sleep(config.MAIN_LOOP_INTERVAL)
    
    return was_armed

def cleanup(camera, pixhawk, metrics, was_armed, logger):
    """Clean up resources before exit"""
    logger.info("=" * 60)
    logger.info("⚠ INITIATING SHUTDOWN ⚠")
    logger.info("=" * 60)
    logger.info(">>> Stopping mission tasks...")

    if was_armed:
        logger.info(">>> Finalizing flight metrics...")
        metrics.end_flight()
        
        # Generate flight summary JSON
        total_waypoints = pixhawk.get_last_waypoint() if pixhawk else 0
        logger.info(">>> Generating flight summary...")
        summary_path = uploader.finalize_flight_summary(metrics.flight_id, total_waypoints)
        if summary_path:
            logger.info(f"✓ Flight summary created: {summary_path}")

    # Cleanup camera
    try:
        camera.close()
    except Exception as e:
        logger.info(f"⚠ Error closing camera: {e} ⚠")

    # Cleanup pixhawk
    try:
        if pixhawk and pixhawk.master:
            pixhawk.master.close()
            logger.info("✓ Pixhawk connection closed.")
    except Exception as e:
        logger.warning(f"⚠ Error closing Pixhawk: {e} ⚠")
    
    uploader.stop_upload_queue()
    logger.info("✓ Shutdown complete.")

def main():
    global running
    
    logger = setup_logging()
    logger.setLevel(logging.INFO)
    
    uploader.start_upload_queue()
    pixhawk = Pixhawk()
    camera = Camera()

    logger.info(">>> Initializing AI detector...")
    try:
        classifier = PinyaSuriAI()
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize AI detector: {e}")
        logger.error("   System will continue without AI detection")
        classifier = None

    next_flight_number = get_next_daily_flight_number()
    metrics = FlightMetricsLogger(flight_number=next_flight_number)

    logger.info("=" * 60)
    logger.info("🍍 PINYASURI FLIGHT SYSTEM 🚁")
    logger.info("=" * 60)
    
    # Wait for connection
    try:
        pixhawk.wait_for_connection()
        pixhawk.request_mission_count()
        pixhawk.request_mission_waypoints()
        initialize_image_log()
        initialize_detection_csv()
        was_armed = main_loop(pixhawk, camera, classifier, metrics, logger)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("⚠ MANUAL STOP - Interrupted by user! ⚠")
        logger.info("=" * 60)

        time.sleep(0.5)
        was_armed = pixhawk.armed if pixhawk else False
        
    except Exception as e:
        logger.error(f"⚠ Fatal error: {e} ⚠", exc_info=True)
        was_armed = False
        
    finally:
        cleanup(camera, pixhawk, metrics, was_armed, logger)

if __name__ == "__main__":
    main()