#!/usr/bin/env python3
# pixhawk.py

import time
import logging
import math
from collections import deque
import config
from pymavlink import mavutil

logger = logging.getLogger(__name__)

class Pixhawk:
    def __init__(self):
        self.master = mavutil.mavlink_connection(config.PIXHAWK_ADDRESS, baud=57600)

        # Mission tracking
        self.mission_count = None  # Total number of waypoints in mission
        self.last_wp = None

        # Flight State
        self.last_wp = None
        self.position = None
        self.armed = False
        self.mode = "UNKNOWN"

        # Sensors
        self.imu_accel = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.groundspeed = 0.0

        # Attitude Data
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Telemetry
        self.last_msg_time = None
        
        # Distance to current waypoint
        self.wp_dist = None 

        # Altitude Stability Tracking
        self.altitude_history = deque(maxlen=10)

        # Waypoint Data
        self.current_waypoint = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.nav_state = "UNKNOWN"
        

    # ---------------------------------------------------------
    # CONNECTION & STREAM SETUP
    # ---------------------------------------------------------
    def wait_for_connection(self):
        logger.info(">>> Waiting for heartbeat...")
        self.master.wait_heartbeat()
        logger.info("✓ Pixhawk connected successfully!")
        self._request_required_streams()
        logger.info("✓ MAVLink streams configured!")
    
    def _request_message(self, msg_id, rate_hz):
        """Request a MAVLink message at a specific rate"""
        interval_us = int(1e6 / rate_hz)

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us, 
            0, 0, 0, 0, 0
        )

        time.sleep(0.05)

    def _request_required_streams(self):
        logger.info(">>> Requesting MAVLink message streams...")

        # Core System State
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT, 1)

        # Mission/Waypoints
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_MISSION_CURRENT, 10)
        
        # Request navigation controller output for wp_dist
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_NAV_CONTROLLER_OUTPUT, 10)

        # Position & Altitude
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10)

        # Attitude
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50)

        # Battery
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 2)
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 2)

        # IMU Data
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_RAW_IMU, 5)

    # ---------------------------------------------------------
    # MISSION COUNTING
    # ---------------------------------------------------------
    def request_mission_count(self):
        """Request the total number of waypoints in the current mission and cache their coordinates"""
        self.master.mav.mission_request_list_send(
            self.master.target_system,
            self.master.target_component
        )
        
        # Wait for response
        msg = self.master.recv_match(type='MISSION_COUNT', blocking=True, timeout=5)
        if msg:
            self.mission_count = msg.count
            logger.info(f"✓ Mission has {self.mission_count} waypoints (0 to {self.mission_count - 1})")
            return self.mission_count
        else:
            logger.warning("⚠ Failed to get mission count")
            return None

    def get_last_waypoint(self):
        """Get the last waypoint number (mission_count - 1)"""
        if self.mission_count is not None:
            return self.mission_count - 1
        return None

    def request_mission_waypoints(self):
        """Request all waypoints and cache their coordinates."""
        if self.mission_count is None:
            self.request_mission_count()
        
        self.mission_waypoints = []
        for i in range(self.mission_count):
            self.master.mav.mission_request_send(
                self.master.target_system,
                self.master.target_component,
                i
            )
            msg = self.master.recv_match(type='MISSION_ITEM', blocking=True, timeout=5)
            if msg:
                self.mission_waypoints.append({
                    "lat": msg.x,
                    "lon": msg.y,
                    "alt": msg.z
                })

    # ---------------------------------------------------------
    # TELEMETRY UPDATE LOOP
    # ---------------------------------------------------------    
    def update(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                break

            self.last_msg_time = time.time()
            msg_type = msg.get_type()

            # -------------------------------
            # POSITION
            # -------------------------------
            if msg_type == "GLOBAL_POSITION_INT":
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                
                # Basic sanity check (valid Earth coordinates)
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    self.position = {
                        "lat": lat,
                        "lon": lon,
                        "rel_alt": msg.relative_alt / 1000.0
                    }
                    self.groundspeed = math.sqrt(msg.vx**2 + msg.vy**2) / 100.0
                    self.altitude_history.append(self.position["rel_alt"])
                else:
                    logger.warning(f"Invalid GPS coordinates: lat={lat}, lon={lon}")

            # -------------------------------
            # ATTITUDE (Roll, Pitch, Yaw)
            # -------------------------------
            elif msg_type == "ATTITUDE":
                # Convert from radians to degrees
                self.roll = math.degrees(msg.roll)
                self.pitch = math.degrees(msg.pitch)
                self.yaw = math.degrees(msg.yaw)

            # -------------------------------
            # NAVIGATION CONTROLLER OUTPUT (contains wp_dist)
            # -------------------------------
            if msg_type == "NAV_CONTROLLER_OUTPUT":
                self.wp_dist = msg.wp_dist
                logger.debug(f"Distance to WP{self.last_wp}: {self.wp_dist:.2f}m")

            # -------------------------------
            # WAYPOINT (Current)
            # -------------------------------
            elif msg_type == "MISSION_CURRENT":
                new_wp = msg.seq
                if 0 <= new_wp <= 255:
                    if self.last_wp != new_wp:
                        logger.debug(f"> Waypoint changed: {self.last_wp} -> {new_wp}")
                        self.last_wp = new_wp

                        # Fetch waypoint coordinates from mission (if mission cached)
                        if hasattr(self, "mission_waypoints") and new_wp < len(self.mission_waypoints):
                            wp = self.mission_waypoints[new_wp]
                            self.current_waypoint = {
                                "lat": wp["lat"],
                                "lon": wp["lon"],
                                "alt": wp["alt"]
                            }

            # -------------------------------
            # HEARTBEAT (MODE + ARM)
            # -------------------------------
            elif msg_type == "HEARTBEAT":
                self.armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.mode = self.master.flightmode


            # -------------------------------
            # IMU
            # -------------------------------
            elif msg_type == "RAW_IMU":
                self.imu_accel = {
                    "x": msg.xacc / 1000.0 * 9.81,
                    "y": msg.yacc / 1000.0 * 9.81,
                    "z": msg.zacc / 1000.0 * 9.81
                }


    # ---------------------------------------------------------
    # Attitude Getters (for gimbal)
    # ---------------------------------------------------------
    def get_attitude(self):
        """Get current roll, pitch, yaw in degrees"""
        return (self.roll, self.pitch, self.yaw)
    
    def get_roll(self):
        """Get current roll angle in degrees"""
        return self.roll
    
    def get_pitch(self):
        """Get current pitch angle in degrees"""
        return self.pitch

    # ---------------------------------------------------------
    # Stability Detection Methods
    # ---------------------------------------------------------
    def is_hovering(self, threshold=config.HOVER_SPEED_THRESHOLD):
        """Check if drone is hovering (horizontal velocity < threshold m/s)"""
        return self.groundspeed < threshold
    
    def is_altitude_stable(self, threshold=0.5, window_size=7):
        """Check if altitude is stable (not changing rapidly)"""
        if len(self.altitude_history) < window_size:
            return False
        
        # Get recent altitude samples
        recent_altitudes = list(self.altitude_history)[-window_size:]
        
        # Calculate variation (max - min)
        variation = max(recent_altitudes) - min(recent_altitudes)
        
        return variation <= threshold
    
    def get_altitude_variation(self):
        """Get current altitude variation for debugging"""
        if len(self.altitude_history) < 2:
            return 0.0
        
        recent_altitudes = list(self.altitude_history)
        return max(recent_altitudes) - min(recent_altitudes)
    
    def clear_waypoint_log(self):
        """Clear the waypoint log (call when disarmed/new flight)"""
        self.altitude_history.clear()
        logger.debug("✓ Cleared waypoint & altitude history.")


    # ---------------------------------------------------------
    # SAFETY / HEALTH
    # ---------------------------------------------------------
    def telemetry_ok(self, timeout=2.0):
        """Returns False if telemetry is stale"""
        if self.last_msg_time is None:
            return False
        return (time.time() - self.last_msg_time) < timeout