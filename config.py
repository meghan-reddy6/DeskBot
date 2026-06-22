import os

def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return float(default)

def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return int(default)

def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("true", "1", "yes", "t", "y")

# ==========================================
# Deployment Configuration & Runtime Mode
# ==========================================
CAMERA_TYPE = os.getenv("CAMERA_TYPE", "WEBCAM").upper()         # "WEBCAM" or "GSTREAMER"
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "PYTORCH").upper() # "PYTORCH" or "ONNX"
RUN_MODE = os.getenv("RUN_MODE", "GUI").upper()                  # "GUI" or "HEADLESS"

# ==========================================
# Camera & Sensor Pipeline
# ==========================================
CAMERA_INDEX = _get_int("CAMERA_INDEX", 0)
FRAME_WIDTH = _get_int("FRAME_WIDTH", 640)
FRAME_HEIGHT = _get_int("FRAME_HEIGHT", 480)

# ==========================================
# Neural Network Models & Classes
# ==========================================
YOLO_POSE_MODEL = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
YOLO_ONNX_MODEL = os.getenv("YOLO_ONNX_MODEL", "yolov8n-pose.onnx")
BOTTLE_CLASS_ID = _get_int("BOTTLE_CLASS_ID", 39)
CUP_CLASS_ID = _get_int("CUP_CLASS_ID", 41)

# ==========================================
# Smoothing & Stabilizers
# ==========================================
SMOOTHING_WINDOW_SIZE = _get_int("SMOOTHING_WINDOW_SIZE", 5)
POSTURE_CONFIDENCE_THRESHOLD = _get_float("POSTURE_CONFIDENCE_THRESHOLD", 0.40)

# --- Standing Detection Calibration Diagnostics ---
POSE_SHOULDER_MIN_CONF = _get_float("POSE_SHOULDER_MIN_CONF", 0.45)
POSE_NOSE_LOST_CONF = _get_float("POSE_NOSE_LOST_CONF", 0.35)
TORSO_STRAIGHT_THRESHOLD = _get_float("TORSO_STRAIGHT_THRESHOLD", 1.05)
TOP_FRAME_CLIP_BOUNDARY = _get_float("TOP_FRAME_CLIP_BOUNDARY", 0.35)

# ==========================================
# Ergonomics & Biomechanical Thresholds
# ==========================================
MAX_SITTING_TIME_SEC = _get_float("MAX_SITTING_TIME_MIN", 30.0) * 60.0      
STAND_RESET_THRESHOLD_SEC = _get_float("STAND_RESET_THRESHOLD_MIN", 2.0) * 60.0           
SLOUCH_ANGLE_THRESHOLD = _get_float("SLOUCH_ANGLE_THRESHOLD", 145.0)          
EYE_STRAIN_LIMIT_SEC = _get_float("EYE_STRAIN_LIMIT_MIN", 20.0) * 60.0    

# ==========================================
# Hydration Tracking
# ==========================================
HYDRATION_REMINDER_INTERVAL = _get_float("HYDRATION_REMINDER_INTERVAL", 30 * 60.0) 
LOW_WATER_THRESHOLD_PERCENT = _get_float("LOW_WATER_THRESHOLD_PERCENT", 20.0)
STANDARD_VESSEL_CAPACITY_ML = _get_float("STANDARD_VESSEL_CAPACITY_ML", 500.0)   

# ==========================================
# Notifications & Alerts
# ==========================================
ENABLE_VOICE_ALERTS = _get_bool("ENABLE_VOICE_ALERTS", True)
ENABLE_DESKTOP_ALERTS = _get_bool("ENABLE_DESKTOP_ALERTS", True)
NOTIFICATION_COOLDOWN = _get_int("NOTIFICATION_COOLDOWN", 30)