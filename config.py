#!/usr/bin/env python3
# config.py

from pathlib import Path
from datetime import datetime


# ============================================================================  
# BASE DIRECTORIES  
# ============================================================================  
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
IMAGE_DIR = BASE_DIR / "images"
MODEL_DIR = BASE_DIR / "models"
JSON_DIR = BASE_DIR / "results"

def ensure_directories():
    """Ensure all base directories exist"""
    LOG_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    JSON_DIR.mkdir(exist_ok=True)


# ============================================================================  
# FLIGHT LOG FILES  
# ============================================================================  
def get_flight_log_file():
    """Daily flight log file (one per day)"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"flight_{date_str}.log"

# Single CSV files (accumulate all flights)
FLIGHT_RAW_CSV = LOG_DIR / "raw_flight_data.csv"
IMAGE_LOG_CSV = LOG_DIR / "image_captures.csv"
CLASSIFICATION_CSV = LOG_DIR / "ai_classifications.csv"

# Daily flight log (for logging only)
FLIGHT_LOG_FILE = get_flight_log_file()


# ============================================================================  
# IMAGE CAPTURE DIRECTORY (daily subfolders)  
# ============================================================================  
def get_image_day_dir():
    date_str = datetime.utcnow().strftime("%Y%m%d")  # YYYYMMDD
    day_folder = IMAGE_DIR / date_str
    day_folder.mkdir(parents=True, exist_ok=True)
    return day_folder


# ============================================================================  
# PIXHAWK CONNECTION SETTINGS  
# ============================================================================  
PIXHAWK_ADDRESS = "/dev/ttyAMA0"


# ============================================================================  
# MISSION WAYPOINT DEFINITIONS  
# ============================================================================  
WP_HOME = 0
WP_TAKEOFF = 1

def is_mapping_waypoint(wp_number, last_wp=None):
    excluded = [WP_HOME, WP_TAKEOFF]
    
    # Exclude last waypoint if provided
    if last_wp is not None:
        excluded.append(last_wp)
    
    return wp_number not in excluded

def get_waypoint_name(wp_number):
    if wp_number == WP_HOME:
        return "HOME"
    elif wp_number == WP_TAKEOFF:
        return "TAKEOFF"
    else:
        return f"WAYPOINT_{wp_number}"

def get_waypoint_type(wp_number):
    if wp_number == WP_HOME:
        return "home"
    elif wp_number == WP_TAKEOFF:
        return "takeoff"
    elif is_mapping_waypoint(wp_number):
        return "waypoint"
    else:
        return "other"


# ============================================================================  
# FLIGHT CAPTURE CONFIGURATION  
# ============================================================================  
MAIN_LOOP_INTERVAL = 0.05  # seconds
MIN_ALTITUDE_FOR_CAPTURE = 0.5  # meters - minimum altitude (safety floor)
MAX_ALTITUDE_FOR_CAPTURE = 5  # meters - maximum altitude (upper limit)
WAYPOINT_CAPTURE_DISTANCE = 1.5  # meters - trigger capture when within this distance
HOVER_SPEED_THRESHOLD = 0.5 # m/s
STABILIZATION_DELAY = 2  # seconds


# ============================================================================  
# BURST CAPTURE CONFIGURATION  
# ============================================================================  
BURST_CAPTURE_COUNT = 5  # Number of images per waypoint
BURST_INTERVAL = 0.5  # Seconds between captures


# ============================================================================  
# AI CONFIGURATION  
# ============================================================================  
SERVER = "http://WEB_SERVER_IP:5000"
MODEL_PATH = MODEL_DIR / "YOLOv8n_PinyaSuri_AI.tflite"
DETECTION_THRESHOLD = 0.5

# Non-Maximum Suppression threshold
NMS_IOU_THRESHOLD = 0.5  # Remove overlapping boxes with IoU > 0.5

# Visualization settings
DRAW_BBOXES = True  # Draw bounding boxes on saved images
BBOX_THICKNESS = 2
FONT_SCALE = 0.6

# Class names - MUST match your training labels in exact order
CLASS_NAMES = {
    0: "Crown Rot Disease",
    1: "Fruit Fasciation Disorder",
    2: "Fruit Rot Disease",
    3: "Healthy",
    4: "Mealybug Wilt Disease",
    5: "Multiple Crown Disorder",
    6: "Root Rot Disease"
}

# Class colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 139),      # Crown Rot - Dark Red
    1: (255, 0, 255),    # Fruit Fasciation - Magenta
    2: (0, 0, 255),      # Fruit Rot - Red
    3: (0, 255, 0),      # Healthy - Green
    4: (0, 165, 255),    # Mealybug Wilt - Orange
    5: (128, 0, 128),    # Multiple Crown - Purple
    6: (0, 0, 255)       # Root Rot - Red
}

def get_class_name(index: int, class_names: dict = None) -> str:
    names = class_names if class_names is not None else CLASS_NAMES
    return names.get(index, f"unknown_{index}")

def get_class_color(index: int) -> tuple:
    """Get BGR color tuple for a class index"""
    return CLASS_COLORS.get(index, (255, 255, 255))  # White for unknown