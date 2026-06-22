import time
import cv2
import logging

import config
from camera_stream import CrossPlatformCapture
from pose_detector import UnifiedEdgeDetector
from hydration_manager import HydrationManager
from notification_manager import NotificationManager
from utils import TelemetryLogger, draw_dashboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing DeskBot Daemon...")
    
    # 1. Environment & Config Assessment
    is_headless = (config.RUN_MODE == "HEADLESS")
    logger.info(f"Running in {config.RUN_MODE} mode with {config.INFERENCE_BACKEND} backend.")
    logger.info(f"Camera Pipeline: {config.CAMERA_TYPE}")
    
    # Instantiate modular managers
    telemetry = TelemetryLogger('deskbot_metrics.db')
    alert_notifier = NotificationManager()
    hydration_handler = HydrationManager(alert_notifier)
    
    # Assign correct model payload based on runtime configurations
    model_path = config.YOLO_ONNX_MODEL if config.INFERENCE_BACKEND == "ONNX" else config.YOLO_POSE_MODEL
    
    detector = UnifiedEdgeDetector(
        backend=config.INFERENCE_BACKEND, 
        model_path=model_path, 
        headless=is_headless
    )
    
    # 2. Lifecycle Orchestration
    # Using robust try/finally and context managers to guarantee clean shutdown
    try:
        telemetry.start()
        
        # CrossPlatformCapture manages its own background thread and cleans up on __exit__
        with CrossPlatformCapture(camera_index=config.CAMERA_INDEX, 
                                  width=config.FRAME_WIDTH, 
                                  height=config.FRAME_HEIGHT) as camera:
                                  
            # State Trackers
            sitting_epoch = time.time()
            standing_epoch = None
            accumulated_sitting_sec = 0
            accumulated_standing_sec = 0
            accumulated_eye_strain_sec = 0
            
            is_currently_slouching = False
            break_completion_alert_fired = False
            
            last_db_commit_timestamp = time.time()
            previous_frame_timestamp = time.time()
            
            # Calibration Variables
            calibration_start = time.time()
            calibration_duration = 4.0
            
            logger.info("Starting continuous inference loop...")
            if not is_headless:
                logger.info("Press 'q' in the OpenCV window to exit gracefully.")
            
            while True:
                # Ingest frames synchronously from the bounceless background thread queue
                ret, raw_frame = camera.read()
                if not ret:
                    logger.warning("Stream disconnected or frame dropped.")
                    time.sleep(0.1)
                    continue
                    
                loop_start = time.time()
                loop_duration = loop_start - previous_frame_timestamp
                previous_frame_timestamp = loop_start
                
                # --- INFERENCE PIPELINE ---
                # Detector guarantees uniformly structured metrics dictionaries
                metrics, annotated_frame = detector.process_frame(raw_frame)
                
                # --- STARTUP CALIBRATION PHASE ---
                if not detector.calibrated:
                    if time.time() - calibration_start < calibration_duration:
                        # Provide visual feedback during calibration if running via GUI
                        if not is_headless and annotated_frame is not None:
                            cv2.putText(annotated_frame, "CALIBRATING POSTURE: SIT STRAIGHT", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.imshow("DeskBot Workspace Monitor", annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        continue # Skip accumulator evaluation during hardware warmup
                    else:
                        # Safe division fallbacks for array processing
                        if len(detector.calib_aspect_ratio) > 10:
                            detector.base_aspect_ratio = sum(detector.calib_aspect_ratio) / len(detector.calib_aspect_ratio)
                            detector.base_Sb = sum(detector.calib_Sb) / len(detector.calib_Sb)
                            detector.base_centroid_y = sum(detector.calib_centroid_y) / len(detector.calib_centroid_y)
                            detector.base_torso_ratio = sum(detector.calib_torso_ratio) / len(detector.calib_torso_ratio)
                            detector.base_nose_to_box = sum(detector.calib_nose_to_box) / len(detector.calib_nose_to_box)
                        else:
                            logger.warning("Calibration data sparse! Utilizing structural fallback baselines.")
                            detector.base_aspect_ratio = 1.2
                            detector.base_Sb = 150.0
                            detector.base_centroid_y = config.FRAME_HEIGHT / 2.0
                            detector.base_torso_ratio = 0.75
                            detector.base_nose_to_box = 50.0

                        detector.calibrated = True
                        logger.info("Calibration Successful! Dual-Mode Architecture Locked.")
                        # Proceed into continuous execution
                            
                # --- STATE CALCULATION MACHINE ---
                current_epoch = time.time()
                
                if metrics["is_sitting"]:
                    standing_epoch = None
                    accumulated_standing_sec = 0
                    break_completion_alert_fired = False
                    
                    accumulated_sitting_sec = current_epoch - sitting_epoch
                    
                    # Core Postural Alert Dispatches
                    if accumulated_sitting_sec > config.MAX_SITTING_TIME_SEC:
                        alert_notifier.send_alert("posture", "Posture Alert", "Prolonged sitting detected. Please stand up!")
                        
                    if metrics["is_slouching"]:
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
                    
                    if accumulated_standing_sec >= config.STAND_RESET_THRESHOLD_SEC:
                        if not break_completion_alert_fired:
                            alert_notifier.send_alert(
                                "break_complete", 
                                "Break Completed", 
                                "Excellent job! You have stood long enough. You can sit down now."
                            )
                            break_completion_alert_fired = True
                        
                        # Once reset condition satisfied, wipe accumulated sitting fatigue
                        sitting_epoch = current_epoch
                        accumulated_sitting_sec = 0

                # --- EYE STRAIN MACHINE ---
                if metrics["is_sitting"] and metrics["is_gazing_screen"]:
                    accumulated_eye_strain_sec += loop_duration
                    if accumulated_eye_strain_sec > config.EYE_STRAIN_LIMIT_SEC:
                        alert_notifier.send_alert("eye_strain", "Eye Fatigue Break", "Screen viewing limit reached.")
                else:
                    # Natural decay mechanism when looking away from the monitor
                    accumulated_eye_strain_sec = max(0, accumulated_eye_strain_sec - (loop_duration * 1.5))

                # --- HYDRATION DISPATCH ---
                hydration_handler.update(
                    metrics["is_drinking"], 
                    metrics["vessel_detected"], 
                    metrics["water_level_pct"]
                )

                # --- ASYNC TELEMETRY PUSH ---
                if current_epoch - last_db_commit_timestamp > 5.0:
                    # Non-blocking enqueue. Worker batch-inserts it onto flash later.
                    telemetry.log(
                        accumulated_sitting_sec,
                        metrics["is_slouching"],
                        accumulated_eye_strain_sec,
                        metrics["water_level_pct"],
                        metrics["current_volume_ml"]
                    )
                    last_db_commit_timestamp = current_epoch

                # --- HEADLESS / GUI EXECUTION BRANCHING ---
                if not is_headless and annotated_frame is not None:
                    # 3. Compute overlay vectors and render to system compositor
                    calculated_fps = 1.0 / loop_duration if loop_duration > 0 else 0.0
                    display_frame = draw_dashboard(
                        annotated_frame, 
                        accumulated_sitting_sec, 
                        accumulated_eye_strain_sec,
                        metrics["is_sitting"], 
                        metrics["is_slouching"],
                        metrics["vessel_type"], 
                        metrics["water_level_pct"],
                        metrics["current_volume_ml"], 
                        calculated_fps
                    )
                    
                    # Status warnings are now exclusively managed by the hardware LED rendering module in utils.py

                    cv2.imshow("DeskBot Workspace Monitor", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): 
                        logger.info("Keyboard interrupt received via GUI.")
                        break

    except KeyboardInterrupt:
        logger.info("Shutdown signal received via SIGINT/Ctrl+C.")
    except Exception as e:
        logger.error(f"Fatal anomaly in core execution loop: {e}", exc_info=True)
    finally:
        # Guarantee memory release and port cleanup regardless of execution success
        logger.info("Initiating graceful shutdown orchestration...")
        telemetry.stop()
        alert_notifier.shutdown()
        if not is_headless:
            cv2.destroyAllWindows()
        logger.info("Daemon shutdown complete. Resources cleanly released.")

if __name__ == "__main__":
    main()