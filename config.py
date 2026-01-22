# ============================================================================  
# AI CONFIGURATION  
# ============================================================================  
SERVER = "http://WEB_SERVER_IP:5000"
MODEL_PATH = MODEL_DIR / "YOLOv8n_PinyaSuri_AI.tflite"
DETECTION_THRESHOLD = 0.5

# Non-Maximum Suppression threshold
NMS_IOU_THRESHOLD = 0.5  # Remove overlapping boxes with IoU > 0.5

# Visualization settings (for future use)
DRAW_BBOXES = True  # Draw bounding boxes on saved images
BBOX_THICKNESS = 2
FONT_SCALE = 0.6

# Class names - MUST match your training labels in exact order
DEFAULT_CLASS_NAMES = {
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
    """
    Get class name from index
    
    Args:
        index: Class index
        class_names: Optional dictionary of class names (from model metadata)
                    If None, uses DEFAULT_CLASS_NAMES
    
    Returns:
        Class name string
    """
    names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    return names.get(index, f"unknown_{index}")

def get_class_color(index: int) -> tuple:
    """Get BGR color tuple for a class index"""
    return CLASS_COLORS.get(index, (255, 255, 255))