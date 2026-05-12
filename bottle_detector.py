import cv2
import numpy as np
from ultralytics import YOLO
import config

class BottleDetector:
    def __init__(self):
        # Loads YOLOv8 Nano model. Will download automatically on first run.
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.classes_to_detect = [config.BOTTLE_CLASS_ID, config.CUP_CLASS_ID]

    def estimate_water_level(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        horizontal_projection = np.sum(edges, axis=1)
        height = horizontal_projection.shape[0]
        margin = int(height * 0.1) 
        
        if height - margin <= margin:
            return 50.0
            
        valid_projection = horizontal_projection[margin:height-margin]
        if len(valid_projection) == 0 or np.max(valid_projection) == 0:
            return 0.0

        water_line_y = np.argmax(valid_projection) + margin
        fill_ratio = (height - water_line_y) / height
        return min(max(fill_ratio * 100, 0), 100)

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        
        bottle_detected = False
        water_level = None
        best_bbox = None
        max_conf = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in self.classes_to_detect and conf > 0.5:
                if conf > max_conf:
                    max_conf = conf
                    best_bbox = [int(i) for i in box.xyxy[0]]
                    bottle_detected = True

        if bottle_detected:
            x1, y1, x2, y2 = best_bbox
            water_level = self.estimate_water_level(frame, best_bbox)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Bottle: {water_level:.0f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame, bottle_detected, water_level