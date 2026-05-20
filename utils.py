import cv2
import sqlite3

def setup_telemetry_db():
    """Initializes local SQLite server instance directly on the board file system."""
    conn = sqlite3.connect('deskbot_metrics.db')
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
    conn.close()

def append_telemetry(sitting_time, slouching_flag, eye_strain_time, water_pct, fluid_ml):
    """Saves records safely completely offline."""
    try:
        conn = sqlite3.connect('deskbot_metrics.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (sitting_duration_seconds, slouch_status, eye_strain_duration_seconds, fluid_level_percent, fluid_volume_ml)
            VALUES (?, ?, ?, ?, ?)
        ''', (sitting_time, 1 if slouching_flag else 0, eye_strain_time, water_pct if water_pct else 0.0, fluid_ml))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database sync fault occurred: {e}")

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