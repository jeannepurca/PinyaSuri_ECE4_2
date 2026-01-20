#!/usr/bin/env python3
# main.py - DEBUGGING VERSION

import time
import csv
import logging
import sys
from pathlib import Path
import config
from logging_config import setup_logging
from pixhawk import Pixhawk
from camera import Camera
from metrics import get_next_daily_flight_number
from metrics import FlightMetricsLogger

running = True

# ----------------------------
# CSV Initialization
# ----------------------------
def initialize_csv():
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
                "image_path"
            ])

def log_image_capture(flight_id, flight_number, waypoint, position, burst_id, burst_index, image_path, logger):
    """Append image capture record to CSV with error handling"""
    try:
        with open(config.IMAGE_LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
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
                image_path
            ])
    except Exception as e:
        logger.error(f"Failed to log image capture to CSV: {e}")

# ----------------------------
# Capture handler
# ----------------------------
def handle_waypoint_capture(pixhawk, camera, metrics, waypoint, flight_number, captured_wp, logger):
    """Capture burst images at waypoint - FAST VERSION with mode change detection"""
    if waypoint in captured_wp:
        return False
        
    wp_name = config.get_waypoint_name(waypoint)
    logger.info("=" * 60)
    logger.info(f">>> {wp_name} (WP{waypoint}) REACHED - Capturing burst images...")
    logger.info("=" * 60)
    
    # Wait for drone to fully stabilize WITH MODE CHECKING
    logger.info(f"⏳ Waiting {config.STABILIZATION_DELAY}s for drone to settle...")
    stabilization_start = time.time()
    
    while (time.time() - stabilization_start) < config.STABILIZATION_DELAY:
        pixhawk.update()  # Keep updating telemetry
        
        # Check if mode changed from AUTO
        if pixhawk.mode != "AUTO":
            logger.warning("=" * 60)
            logger.warning(f"⚠️ CAPTURE ABORTED - Mode changed to {pixhawk.mode}")
            logger.warning("=" * 60)
            return False
        
        time.sleep(0.1)  # Check every 100ms

    pixhawk.update()  # Get latest data after stabilization
    
    # Final mode check before starting burst
    if pixhawk.mode != "AUTO":
        logger.warning("=" * 60)
        logger.warning(f"⚠️ CAPTURE ABORTED - Not in AUTO mode ({pixhawk.mode})")
        logger.warning("=" * 60)
        return False
    
    # Burst capture
    num_captures = config.BURST_CAPTURE_COUNT
    burst_interval = config.BURST_INTERVAL
    captured_images = []
    burst_id = f"{metrics.flight_id}_wp{waypoint}_{int(time.time())}"

    logger.info(f"📸 Starting burst capture ({num_captures} frames)...")

    for i in range(num_captures):
        # Check mode before each capture
        pixhawk.update()
        if pixhawk.mode != "AUTO":
            logger.warning("=" * 60)
            logger.warning(f"⚠️ BURST ABORTED at frame {i+1}/{num_captures} - Mode changed to {pixhawk.mode}")
            logger.warning(f"   Captured {len(captured_images)} images before abort")
            logger.warning("=" * 60)
            
            # Mark as captured if we got at least one image
            if captured_images:
                captured_wp.add(waypoint)
            return len(captured_images) > 0
        
        try:
            image_path = camera.capture(
                waypoint=waypoint,
                flight_number=flight_number,
                prefix="pinyasuri",
                burst_index=i
            )
            
            # Verify file
            if Path(image_path).exists() and Path(image_path).stat().st_size > 1000:
                captured_images.append(image_path)
                
                # Log with burst metadata
                log_image_capture(
                    metrics.flight_id, 
                    flight_number, 
                    waypoint, 
                    pixhawk.position,
                    burst_id,
                    i,
                    image_path, 
                    logger
                )
                
                logger.info(f"  ✓ Frame {i+1}/{num_captures} captured "
                           f"(size: {Path(image_path).stat().st_size / 1024:.1f} KB)")
            else:
                logger.error(f"⚠ Frame {i+1} file invalid or too small")
            
            if i < num_captures - 1:
                time.sleep(burst_interval)
                
        except Exception as e:
            logger.error(f"⚠ Burst frame {i+1} failed: {e}")

    if captured_images:
        logger.info(f"✓ CAPTURED {len(captured_images)}/{num_captures} images at {wp_name}")
        logger.info(f"  Burst ID: {burst_id}")
        captured_wp.add(waypoint)
        return True
    else:
        logger.error("⚠ Burst capture completely failed - no valid images")
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
        "battery_remaining": pixhawk.battery_remaining
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
    logger.info(f"✅ ALL CHECKS PASSED - Triggering capture for WP{waypoint}!")
    logger.info(f"   Altitude: {pixhawk.position['rel_alt']:.2f}m")
    logger.info(f"   Distance to waypoint: {pixhawk.wp_dist:.2f}m")
    logger.info(f"   Groundspeed: {pixhawk.groundspeed:.2f} m/s")
    logger.info("=" * 60)
    return True


def main_loop(pixhawk, camera, metrics, logger):
    captured_wp = set()
    flight_number = metrics.flight_number
    was_armed = False
    current_mode = "UNKNOWN"
    current_waypoint = None  # ✅ Initialize here at the start
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
            # Show distance instead of reached log
            dist_str = f"{pixhawk.wp_dist:.2f}m" if pixhawk.wp_dist else "N/A"
            logger.debug(f"[STATUS] Mode: {pixhawk.mode}, WP: {pixhawk.last_wp}, "
                        f"Alt: {pixhawk.position['rel_alt']:.1f}m, "
                        f"Dist to WP: {dist_str}, "
                        f"Captured: {captured_wp}")

        # Update metrics during flight
        if pixhawk.armed:
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
                "flight_mode": pixhawk.mode,
                "is_hovering": pixhawk.is_hovering(threshold=1.0),
                "battery": {
                    "voltage": pixhawk.battery_voltage,
                    "current": pixhawk.battery_current,
                    "percentage": pixhawk.battery_remaining
                }
            }
            metrics.log_telemetry(telemetry)

        
        # Check for image capture
        if should_capture_image(pixhawk, pixhawk.last_wp, captured_wp, logger):
            handle_waypoint_capture(
                pixhawk, camera, metrics, 
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
    
    logger.info("✓ Shutdown complete.")

def main():
    global running
    
    logger = setup_logging()
    
    # SET DEBUG LEVEL
    logger.setLevel(logging.INFO)
    
    pixhawk = Pixhawk()
    camera = Camera()
    next_flight_number = get_next_daily_flight_number()
    metrics = FlightMetricsLogger(flight_number=next_flight_number)

    logger.info("=" * 60)
    logger.info("🍍 PINYASURI FLIGHT SYSTEM 🚁")
    logger.info("=" * 60)
    
    # Wait for connection
    try:
        pixhawk.wait_for_connection()
        pixhawk.request_mission_count()
        initialize_csv()
        was_armed = main_loop(pixhawk, camera, metrics, logger)
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