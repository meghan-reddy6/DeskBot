import cv2
import time
import config
from pose_detector import PoseDetector
from bottle_detector import BottleDetector
from hydration_manager import HydrationManager
from notification_manager import NotificationManager
from utils import draw_dashboard

def main():
    print("Initializing Desk Wellness Bot...")
    
    # Initialize hardware and modules
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    pose_det = PoseDetector()
    bottle_det = BottleDetector()
    notifier = NotificationManager()
    hydro_mgr = HydrationManager(notifier)

    # State variables
    sitting_start_time = time.time()
    standing_start_time = None
    total_sitting_time = 0
    pTime = 0

    print("System Ready. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process Pose
        frame, is_sitting, is_drinking, _ = pose_det.process_frame(frame)

        # Process Bottle (YOLO)
        frame, bottle_detected, water_level = bottle_det.process_frame(frame)

        # Update Hydration Logic
        time_since_drink = hydro_mgr.update(is_drinking, bottle_detected, water_level)

        # Update Sitting Logic
        current_time = time.time()
        if is_sitting:
            if standing_start_time is not None:
                standing_start_time = None # Cancel standing timer
            total_sitting_time = current_time - sitting_start_time

            # Trigger prolonged sitting alert
            if total_sitting_time > config.SITTING_TIME_THRESHOLD:
                notifier.send_alert(
                    "posture", 
                    "Posture Check", 
                    "You've been sitting for a long time. Please stand up and stretch!"
                )
        else:
            if standing_start_time is None:
                standing_start_time = current_time
            
            # Reset sitting timer if they've been standing/away long enough
            if current_time - standing_start_time > config.STAND_RESET_THRESHOLD:
                sitting_start_time = current_time
                total_sitting_time = 0

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        # Draw UI
        frame = draw_dashboard(
            frame, total_sitting_time, time_since_drink, 
            is_sitting, water_level, fps
        )

        # Display
        cv2.imshow("Desk Wellness Bot (Debug UI)", frame)

        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()