import cv2
import time
import threading
import numpy as np
import config
from pose_detector import UnifiedEdgeDetector
from hydration_manager import HydrationManager
from notification_manager import NotificationManager
from utils import setup_telemetry_db, append_telemetry, draw_dashboard

# Memory Boundary Locks
thread_lock = threading.Lock()
frame_for_ai = None
shared_overlay_elements = []

network_telemetry = {
    "is_sitting": False,
    "is_slouching": False,
    "is_drinking": False,
    "is_gazing_screen": False,
    "vessel_detected": False,
    "vessel_type": "None",
    "water_level_pct": None,
    "current_volume_ml": 0.0
}

def asynchronous_inference_worker():
    """Independent inference thread. Frees up camera display thread entirely."""
    global frame_for_ai, shared_overlay_elements, network_telemetry
    engine = UnifiedEdgeDetector()
    
    calibration_ratios = []
    calibration_noses = []
    calibration_start = time.time()
    print("[DeskBot] Calibrating posture framework... Please sit completely straight up.")
    
    while True:
        local_frame = None
        with thread_lock:
            if frame_for_ai is not None:
                local_frame = frame_for_ai.copy()
                
        if local_frame is not None:
            (sit, slouch, drink, gaze, v_det, v_type, pct, ml, overlays) = engine.process_frame(local_frame)
            
            if time.time() - calibration_start < 4.0:
                if engine.neck_ratio_history and engine.nose_y_history:
                    calibration_ratios.append(engine.neck_ratio_history[-1])
                    calibration_noses.append(engine.nose_y_history[-1])
                continue
            elif not engine.calibrated and calibration_ratios:
                engine.baseline_neck_ratio = float(np.mean(calibration_ratios))
                engine.baseline_nose_y = float(np.mean(calibration_noses))
                engine.calibrated = True
                print("[DeskBot] Calibration Successful!")

            with thread_lock:
                shared_overlay_elements = overlays
                network_telemetry["is_sitting"] = sit
                network_telemetry["is_slouching"] = slouch
                network_telemetry["is_drinking"] = drink
                network_telemetry["is_gazing_screen"] = gaze
                network_telemetry["vessel_detected"] = v_det
                network_telemetry["vessel_type"] = v_type
                network_telemetry["water_level_pct"] = pct
                network_telemetry["current_volume_ml"] = ml
                
        time.sleep(config.AI_INFERENCE_INTERVAL)

def main():
    global frame_for_ai, shared_overlay_elements, network_telemetry
    print("[DeskBot Daemon] Initialization sequence started...")
    
    setup_telemetry_db()
    video_capture = cv2.VideoCapture(config.CAMERA_INDEX)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    alert_notifier = NotificationManager()
    hydration_handler = HydrationManager(alert_notifier)
    
    background_processor = threading.Thread(target=asynchronous_inference_worker, daemon=True)
    background_processor.start()
    
    sitting_epoch = time.time()
    standing_epoch = None
    accumulated_sitting_sec = 0
    accumulated_eye_strain_sec = 0
    
    previous_frame_timestamp = time.time()
    last_db_commit_timestamp = time.time()
    
    while video_capture.isOpened():
        success, current_frame = video_capture.read()
        if not success: 
            break
            
        loop_duration = time.time() - previous_frame_timestamp
        previous_frame_timestamp = time.time()
        current_epoch = time.time()

        with thread_lock:
            frame_for_ai = current_frame.copy()
            active_metrics = network_telemetry.copy()
            local_overlays = list(shared_overlay_elements)

        for item in local_overlays:
            if item[0] == "circle":
                cv2.circle(current_frame, item[1], item[2], item[3], item[4])
            elif item[0] == "rect":
                cv2.rectangle(current_frame, item[1], item[2], item[3], item[4])

        if active_metrics["is_sitting"]:
            standing_epoch = None
            accumulated_sitting_sec = current_epoch - sitting_epoch
            if accumulated_sitting_sec > config.SITTING_TIME_THRESHOLD:
                alert_notifier.send_alert("posture", "Posture Alert", "Prolonged sitting detected.")
            if active_metrics["is_slouching"]:
                alert_notifier.send_alert("slouch", "Ergonomics Check", "Slouching behavior detected.")
        else:
            if standing_epoch is None: 
                standing_epoch = current_epoch
            if current_epoch - standing_epoch > config.STAND_RESET_THRESHOLD:
                sitting_epoch = current_epoch
                accumulated_sitting_sec = 0

        if active_metrics["is_sitting"] and active_metrics["is_gazing_screen"]:
            accumulated_eye_strain_sec += loop_duration
            if accumulated_eye_strain_sec > config.EYE_STRAIN_THRESHOLD_SEC:
                alert_notifier.send_alert("eye_strain", "Eye Fatigue Break", "Screen viewing limit reached.")
        else:
            accumulated_eye_strain_sec = max(0, accumulated_eye_strain_sec - (loop_duration * 1.5))

        hydration_handler.update(active_metrics["is_drinking"], active_metrics["vessel_detected"], active_metrics["water_level_pct"])

        if current_epoch - last_db_commit_timestamp > 5.0:
            append_telemetry(accumulated_sitting_sec, active_metrics["is_slouching"], accumulated_eye_strain_sec, active_metrics["water_level_pct"], active_metrics["current_volume_ml"])
            last_db_commit_timestamp = current_epoch

        calculated_fps = 1.0 / loop_duration if loop_duration > 0 else 0.0
        current_frame = draw_dashboard(
            current_frame, accumulated_sitting_sec, accumulated_eye_strain_sec,
            active_metrics["is_sitting"], active_metrics["is_slouching"],
            active_metrics["vessel_type"], active_metrics["water_level_pct"],
            active_metrics["current_volume_ml"], calculated_fps
        )

        if active_metrics["is_slouching"]:
            cv2.putText(current_frame, "POOR POSTURE DETECTED", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if accumulated_eye_strain_sec > (config.EYE_STRAIN_THRESHOLD_SEC * 0.75):
            cv2.putText(current_frame, "HIGH EYE FATIGUE RISK", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

        cv2.imshow("DeskBot Workspace Monitor", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()