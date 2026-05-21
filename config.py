# Performance Optimized Configuration for RubikPi 3

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# AI Pipeline
YOLO_POSE_MODEL = "yolov8n-pose.pt" 
BOTTLE_CLASS_ID = 39                
CUP_CLASS_ID = 41                   
AI_INFERENCE_INTERVAL = 0.05        # Drop sleep interval to process faster when thread is free

# Jitter & Glitch Stabilization 
SMOOTHING_WINDOW_SIZE = 5           # History frame count to calculate moving averages
POSTURE_CONFIDENCE_THRESHOLD = 0.40 # Confidence cutoff to prevent flashing skeletons

# Ergonomics & Biomechanical Thresholds
SITTING_TIME_THRESHOLD = 45 * 60      
STAND_RESET_THRESHOLD = 60            
SLOUCH_ANGLE_THRESHOLD = 145          
EYE_STRAIN_THRESHOLD_SEC = 20 * 60    

# Hydration Tracking
HYDRATION_REMINDER_INTERVAL = 30 * 60 
LOW_WATER_THRESHOLD_PERCENT = 20.0
STANDARD_VESSEL_CAPACITY_ML = 500.0   

# Alerts
ENABLE_VOICE_ALERTS = True
ENABLE_DESKTOP_ALERTS = True
NOTIFICATION_COOLDOWN = 10