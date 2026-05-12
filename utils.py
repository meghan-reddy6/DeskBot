import cv2
import time

def draw_dashboard(frame, sitting_time, time_since_drink, is_sitting, water_level, fps):
    """Draws the transparent overlay dashboard on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Status Texts
    sit_status = "Sitting" if is_sitting else "Standing/Away"
    sit_color = (0, 255, 0) if not is_sitting or sitting_time < 2000 else (0, 0, 255)
    
    cv2.putText(frame, f"Posture: {sit_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sit_color, 2)
    cv2.putText(frame, f"Sit Time: {sitting_time // 60:.0f} min", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Time since drink: {time_since_drink // 60:.0f} min", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    wl_text = f"{water_level:.0f}%" if water_level is not None else "N/A"
    cv2.putText(frame, f"Water Level: {wl_text}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame