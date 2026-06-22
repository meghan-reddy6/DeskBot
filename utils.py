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


def draw_dashboard(frame, sitting_time, eye_strain_time, is_sitting, is_slouching, vessel_type, water_level, volume_ml, video_fps):
    """Renders the workspace analytics layer onto the display frame."""
    overlay = frame.copy()
    # Draw transparent backing box for metrics readable UI
    cv2.rectangle(overlay, (10, 10), (380, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Postural UI configuration
    sit_label = "Sitting" if is_sitting else "Standing/Away"
    sit_color = (0, 255, 0) if not is_sitting else ((0, 140, 255) if is_slouching else (255, 255, 0))
    
    cv2.putText(frame, f"User Posture: {sit_label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sit_color, 2)
    cv2.putText(frame, f"Session Duration: {sitting_time // 60:.0f}m {sitting_time % 60:.0f}s", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Eye Fatigue UI Configuration
    from config import EYE_STRAIN_THRESHOLD_SEC
    fatigue_pct = min((eye_strain_time / EYE_STRAIN_THRESHOLD_SEC) * 100, 100)
    cv2.putText(frame, f"Eye Fatigue Index: {fatigue_pct:.0f}% ({eye_strain_time // 60:.0f}m)", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255) if fatigue_pct > 75 else (255, 255, 255), 1)
    
    # Vessel Telemetry
    v_text = f"{vessel_type} ({water_level:.0f}% | {volume_ml:.0f}mL)" if water_level is not None else f"{vessel_type} (N/A)"
    cv2.putText(frame, f"Object Status: {v_text}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Computational Metrics
    cv2.putText(frame, f"System Loop Speed: {video_fps:.1f} FPS", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return frame