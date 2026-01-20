#!/usr/bin/env python3
# metrics.py

import csv
import logging
from datetime import datetime
from pathlib import Path
import config
import fcntl

logger = logging.getLogger(__name__)

# File to persist last flight number per day
DAILY_FLIGHT_FILE = config.LOG_DIR / "daily_flight_id.txt"

def get_next_daily_flight_number():
    """Get next flight number for today; reset if a new day"""
    today_str = datetime.utcnow().strftime("%Y%m%d")
    last_date, last_number = today_str, 0

    # Use file locking to prevent race conditions
    with open(DAILY_FLIGHT_FILE, "a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
            f.seek(0)
            content = f.read().strip()
            
            if content:
                try:
                    last_date_stored, last_number_stored = content.split("-")
                    last_number_stored = int(last_number_stored)
                    last_date, last_number = last_date_stored, last_number_stored
                except Exception:
                    last_date, last_number = today_str, 0
            
            if last_date == today_str:
                next_number = last_number + 1
            else:
                next_number = 1
            
            # Write back
            f.seek(0)
            f.truncate()
            f.write(f"{today_str}-{next_number}")
            
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
    
    return next_number

class FlightMetricsLogger:
    def __init__(self, flight_number: int):
        self.flight_number = flight_number
        self.flight_start_time = None
        self.flight_end_time = None
        self.armed = False

        # Create flight ID using today's date + flight number
        self.flight_id = self._generate_flight_id()

        # CSV setup
        self.csv_file = config.FLIGHT_RAW_CSV
        self._initialize_csv()

    def _generate_flight_id(self):
        date_str = datetime.utcnow().strftime("%Y%m%d")
        return f"{date_str}_F{self.flight_number}"

    def _initialize_csv(self):
        """Create raw flight CSV with header if not existing."""
        try:
            with open(self.csv_file, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "flight_id",
                    "flight_number",
                    "timestamp_utc",
                    "roll_deg", "pitch_deg", "yaw_deg",
                    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                    "lat_deg", "lon_deg", "alt_m",
                    "groundspeed_m_s",
                    "waypoint_index",
                    "waypoint_lat_deg", "waypoint_lon_deg", "waypoint_alt_m",
                    "flight_mode",
                    "is_hovering",
                    "battery_voltage_V",
                    "battery_current_A",
                    "battery_percentage"
                ])
            logger.info("✓ Created raw flight CSV")
        except FileExistsError:
            logger.info("✓ Raw flight CSV already exists")

    # Call when the drone arms
    def start_flight(self):
        self.armed = True
        self.flight_start_time = datetime.utcnow().isoformat()
        self.flight_id = self._generate_flight_id()
        logger.info(f"🛫 Flight {self.flight_id} started at {self.flight_start_time}.")

    # Call when the drone disarms
    def end_flight(self):
        self.armed = False
        self.flight_end_time = datetime.utcnow().isoformat()
        logger.info(f"🛬 Flight {self.flight_id} ended at {self.flight_end_time}.")

        # Increment flight number for next flight
        self.flight_number = get_next_daily_flight_number()
        self.flight_id = self._generate_flight_id()

    def log_telemetry(self, data):
        if not self.armed:
            return

        timestamp = datetime.utcnow().isoformat()

        # Attitude
        roll = data.get("attitude", {}).get("roll", 0.0)
        pitch = data.get("attitude", {}).get("pitch", 0.0)
        yaw = data.get("attitude", {}).get("yaw", 0.0)

        # IMU
        accel_x = data.get("imu_accel", {}).get("x", 0.0)
        accel_y = data.get("imu_accel", {}).get("y", 0.0)
        accel_z = data.get("imu_accel", {}).get("z", 0.0)

        # GPS
        lat = data.get("gps", {}).get("lat", 0.0)
        lon = data.get("gps", {}).get("lon", 0.0)
        alt = data.get("gps", {}).get("alt", 0.0)
        groundspeed = data.get("gps", {}).get("groundspeed", 0.0)

        # Waypoint
        wp_index = data.get("waypoint_index", -1)
        wp_lat = data.get("waypoint_lat", 0.0)
        wp_lon = data.get("waypoint_lon", 0.0)
        wp_alt = data.get("waypoint_alt", 0.0)

        # Flight state
        flight_mode = data.get("flight_mode", "UNKNOWN")

        # Battery - handle None values properly
        battery_voltage = data.get("battery", {}).get("voltage", 0.0) or 0.0
        battery_current = data.get("battery", {}).get("current", 0.0) or 0.0
        battery_percent = data.get("battery", {}).get("percentage") or 0.0

        # Write telemetry row
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.flight_id,
                self.flight_number,
                timestamp,
                round(roll, 3), round(pitch, 3), round(yaw, 3),
                round(accel_x, 3), round(accel_y, 3), round(accel_z, 3),
                round(lat, 7), round(lon, 7), round(alt, 2),
                round(groundspeed, 2),
                wp_index,
                round(wp_lat, 7), round(wp_lon, 7), round(wp_alt, 2),
                flight_mode,
                data.get("is_hovering", False),
                round(battery_voltage, 2),
                round(battery_current, 2),
                round(battery_percent, 1)
            ])