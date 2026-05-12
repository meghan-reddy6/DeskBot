# Configuration settings for the Desk Wellness Bot

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_LIMIT = 30

# Sitting & Posture Settings
SITTING_TIME_THRESHOLD = 45 * 60  # 45 minutes in seconds
STAND_RESET_THRESHOLD = 60        # 60 seconds of standing/absence to reset timer

# Hydration Settings
HYDRATION_REMINDER_INTERVAL = 30 * 60  # 30 minutes in seconds
LOW_WATER_THRESHOLD_PERCENT = 20.0

# Notification Settings
ENABLE_VOICE_ALERTS = True
ENABLE_DESKTOP_ALERTS = True
NOTIFICATION_COOLDOWN = 300  # Minimum 5 minutes between the same type of notification

# YOLOv8 Settings
YOLO_MODEL_PATH = "yolov8n.pt"  # Nano model for fast CPU inference
BOTTLE_CLASS_ID = 39            # COCO dataset class ID for 'bottle'
CUP_CLASS_ID = 41               # COCO dataset class ID for 'cup'