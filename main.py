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
                
                with thread_lock:
                    shared_overlay_elements = overlays
                    network_telemetry["is_sitting"] = True
                    network_telemetry["is_slouching"] = False
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
    
    # Core Engine Core Timestamps
    sitting_epoch = time.time()
    standing_epoch = None
    
    # State Duration Accumulators
    accumulated_sitting_sec = 0
    accumulated_standing_sec = 0
    accumulated_eye_strain_sec = 0
    
    # Condition Latch Flags
    is_currently_slouching = False
    break_completion_alert_fired = False
    
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

        # 1. Render base model overlay landmarks (Skeletons/Bounding Boxes)
        for item in local_overlays:
            if item[0] == "circle":
                cv2.circle(current_frame, item[1], item[2], item[3], item[4])
            elif item[0] == "rect":
                cv2.rectangle(current_frame, item[1], item[2], item[3], item[4])

        # 2. Dual Accumulator Posture State Engine
        if active_metrics["is_sitting"]:
            standing_epoch = None
            accumulated_standing_sec = 0
            break_completion_alert_fired = False
            
            accumulated_sitting_sec = current_epoch - sitting_epoch
            
            if accumulated_sitting_sec > config.SITTING_TIME_THRESHOLD:
                alert_notifier.send_alert("posture", "Posture Alert", "Prolonged sitting detected. Please stand up!")
                
            if active_metrics["is_slouching"]:
                if not is_currently_slouching:
                    alert_notifier.send_alert("slouch", "Ergonomics Check", "Slouching behavior detected.")
                    is_currently_slouching = True
            else:
                is_currently_slouching = False
        else:
            is_currently_slouching = False
            if standing_epoch is None: 
                standing_epoch = current_epoch
                
            accumulated_standing_sec = current_epoch - standing_epoch
            
            if accumulated_standing_sec >= config.STAND_RESET_THRESHOLD:
                if not break_completion_alert_fired:
                    alert_notifier.send_alert(
                        "break_complete", 
                        "Break Completed", 
                        "Excellent job! You have stood long enough. You can sit down now."
                    )
                    break_completion_alert_fired = True
                
                sitting_epoch = current_epoch
                accumulated_sitting_sec = 0

        # 3. Eye Strain Monitoring Engine
        if active_metrics["is_sitting"] and active_metrics["is_gazing_screen"]:
            accumulated_eye_strain_sec += loop_duration
            if accumulated_eye_strain_sec > config.EYE_STRAIN_THRESHOLD_SEC:
                alert_notifier.send_alert("eye_strain", "Eye Fatigue Break", "Screen viewing limit reached.")
        else:
            accumulated_eye_strain_sec = max(0, accumulated_eye_strain_sec - (loop_duration * 1.5))

        hydration_handler.update(active_metrics["is_drinking"], active_metrics["vessel_detected"], active_metrics["water_level_pct"])

        # 4. Telemetry Log Handler
        if current_epoch - last_db_commit_timestamp > 5.0:
            append_telemetry(accumulated_sitting_sec, active_metrics["is_slouching"], accumulated_eye_strain_sec, active_metrics["water_level_pct"], active_metrics["current_volume_ml"])
            last_db_commit_timestamp = current_epoch

        # 5. Clean Dashboard Generator Layer (Passes timers natively to avoid layout collisions)
        calculated_fps = 1.0 / loop_duration if loop_duration > 0 else 0.0
        current_frame = draw_dashboard(
            current_frame, accumulated_sitting_sec, accumulated_eye_strain_sec,
            active_metrics["is_sitting"], active_metrics["is_slouching"],
            active_metrics["vessel_type"], active_metrics["water_level_pct"],
            active_metrics["current_volume_ml"], calculated_fps
        )

        # 6. Integrated Non-Overlapping Status Information Banners (Safe Corner Placements)
        if not active_metrics["is_sitting"]:
            stand_min, stand_sec = divmod(int(accumulated_standing_sec), 60)
            target_min, target_sec = divmod(int(config.STAND_RESET_THRESHOLD), 60)
            color = (0, 255, 0) if break_completion_alert_fired else (0, 255, 255)
            status_str = "BREAK SATISFIED" if break_completion_alert_fired else "STAND BREAK ACTIVE"
            
            # Positioned lower left safely below standard header metrics panel
            cv2.putText(current_frame, f"{status_str}: {stand_min:02d}:{stand_sec:02d}/{target_min:02d}:{target_sec:02d}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Bottom right anchoring for ergonomics alert indicators
        if active_metrics["is_slouching"]:
            cv2.putText(current_frame, "[WARN: POOR POSTURE]", (360, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        elif accumulated_eye_strain_sec > (config.EYE_STRAIN_THRESHOLD_SEC * 0.75):
            cv2.putText(current_frame, "[WARN: EYE STRAIN RISK]", (340, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)

        cv2.imshow("DeskBot Workspace Monitor", current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    video_capture.release()
    cv2.destroyAllWindows()
    alert_notifier.shutdown()

if __name__ == "__main__":
    main()