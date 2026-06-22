import cv2
import sqlite3
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class TelemetryLogger:
    """
    Asynchronous SQLite Worker.
    Maintains an isolated thread handling batched DB writes to prevent I/O blocking
    and minimize flash storage wear on embedded devices.
    """
    def __init__(self, db_path='deskbot_metrics.db'):
        self.db_path = db_path
        # Thread-safe queue for metrics payload buffering
        self.telemetry_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        
        self._setup_db()
        
    def _setup_db(self):
        """Initializes local SQLite server instance table structure."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        sitting_duration_seconds REAL,
                        slouch_status INTEGER,
                        eye_strain_duration_seconds REAL,
                        fluid_level_percent REAL,
                        fluid_volume_ml REAL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize telemetry database: {e}")

    def start(self):
        """Starts the background worker daemon."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Asynchronous telemetry logger started.")

    def stop(self):
        """Safely stops the background worker and drains any pending DB writes."""
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=3.0)
        logger.info("Asynchronous telemetry logger stopped.")

    def log(self, sitting_time, slouching_flag, eye_strain_time, water_pct, fluid_ml):
        """
        Non-blocking function called by the main loop to drop a metric snapshot.
        If the queue backs up, it drops frames rather than blocking the camera.
        """
        payload = (
            sitting_time, 
            1 if slouching_flag else 0, 
            eye_strain_time, 
            water_pct if water_pct is not None else 0.0, 
            fluid_ml
        )
        try:
            # We don't wait; if the queue is overloaded, we discard the telemetry point.
            self.telemetry_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _worker_loop(self):
        """Background daemon loop that holds a single open connection and batches writes."""
        try:
            # Single open connection per thread to minimize I/O overhead
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            while not self.stop_event.is_set():
                batch = []
                try:
                    # Block until at least one item is available
                    item = self.telemetry_queue.get(timeout=1.0)
                    batch.append(item)
                    
                    # Drain the queue to batch writes if items built up behind
                    while not self.telemetry_queue.empty():
                        batch.append(self.telemetry_queue.get_nowait())
                except queue.Empty:
                    continue
                
                if batch:
                    cursor.executemany('''
                        INSERT INTO system_logs (
                            sitting_duration_seconds, 
                            slouch_status, 
                            eye_strain_duration_seconds, 
                            fluid_level_percent, 
                            fluid_volume_ml
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', batch)
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Telemetry worker thread crashed: {e}")
        finally:
            if 'conn' in locals():
                conn.close()


def draw_dashboard(frame, sitting_time, standing_time, eye_strain_time, is_sitting, is_slouching, vessel_type, water_level, volume_ml, video_fps):
    """Renders the workspace analytics layer with a color-coded visual HUD."""
    overlay = frame.copy()
    # Darker, larger backing box for HUD to ensure high contrast
    cv2.rectangle(overlay, (10, 10), (320, 220), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- 1. Posture Status LED ---
    cv2.putText(frame, "Posture:", (20, 40), font, 0.6, (255, 255, 255), 1)
    if not is_sitting:
        color = (0, 255, 255) # True Yellow (BGR)
        status_text = "Standing/Away"
    elif is_slouching:
        color = (0, 140, 255) # Orange Warning
        status_text = "Slouch Warning"
    else:
        color = (0, 255, 0) # Green Good
        status_text = "Good Sitting"
    
    # Draw status LED (filled circle)
    cv2.circle(frame, (120, 35), 8, color, -1)
    cv2.putText(frame, status_text, (140, 40), font, 0.55, color, 1)

    # Session / Break Time indicator directly below posture
    if not is_sitting:
        from config import STAND_RESET_THRESHOLD_SEC
        stand_min, stand_sec = divmod(int(standing_time), 60)
        target_min, target_sec = divmod(int(STAND_RESET_THRESHOLD_SEC), 60)
        if standing_time >= STAND_RESET_THRESHOLD_SEC:
            cv2.putText(frame, f"Break: COMPLETE {stand_min:02d}:{stand_sec:02d}", (20, 70), font, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f"Break: {stand_min:02d}:{stand_sec:02d} / {target_min:02d}:{target_sec:02d}", (20, 70), font, 0.5, color, 1)
    else:
        cv2.putText(frame, f"Session: {sitting_time // 60:.0f}m {sitting_time % 60:.0f}s", (20, 70), font, 0.5, (200, 200, 200), 1)

    # --- 2. Eye Strain Status LED ---
    from config import EYE_STRAIN_LIMIT_SEC
    fatigue_pct = min((eye_strain_time / EYE_STRAIN_LIMIT_SEC) * 100, 100)
    
    cv2.putText(frame, "Eye Strain:", (20, 110), font, 0.6, (255, 255, 255), 1)
    if fatigue_pct < 50:
        e_color = (0, 255, 0) # Green
        e_text = "Neutral"
    elif fatigue_pct < 85:
        e_color = (0, 255, 255) # Yellow Warning
        e_text = "Strain Warning"
    else:
        e_color = (0, 0, 255) # Red Critical
        e_text = "Take a Break"
        
    cv2.circle(frame, (140, 105), 8, e_color, -1)
    cv2.putText(frame, f"{e_text} ({fatigue_pct:.0f}%)", (160, 110), font, 0.55, e_color, 1)

    # --- 3. Hydration Status LED ---
    cv2.putText(frame, "Hydration:", (20, 150), font, 0.6, (255, 255, 255), 1)
    if water_level is None:
        w_color = (150, 150, 150) # Gray
        w_text = "No Vessel"
    else:
        if water_level > 60:
            w_color = (255, 255, 0) # Cyan (Full)
            w_text = f"Full ({water_level:.0f}%)"
        elif water_level > 20:
            w_color = (0, 255, 255) # Yellow
            w_text = f"Half Empty ({water_level:.0f}%)"
        else:
            w_color = (0, 0, 255) # Red Critical
            w_text = f"Refill Required ({water_level:.0f}%)"
            
    cv2.circle(frame, (140, 145), 8, w_color, -1)
    cv2.putText(frame, w_text, (160, 150), font, 0.55, w_color, 1)

    # --- System Performance ---
    cv2.putText(frame, f"System Loop: {video_fps:.1f} FPS", (20, 200), font, 0.45, (0, 255, 0), 1)

    return frame