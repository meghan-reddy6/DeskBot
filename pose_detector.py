import cv2
import math
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self):
        # Load the Pose-specific Nano model (very fast on CPU)
        # It will download 'yolov8n-pose.onnx' automatically on first run
        self.model = YOLO('yolov8n-pose.pt')
        self.baseline_shoulder_y = None

    def process_frame(self, frame):
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        is_sitting = False
        is_drinking = False
        landmarks_dict = {}

        # If we found a person
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            # We take the first person detected
            # Keypoint indices: 0=Nose, 5=L_Shoulder, 6=R_Shoulder, 9=L_Wrist, 10=R_Wrist
            kpts = results.keypoints.xy[0].cpu().numpy() 
            
            # Draw keypoints for debug
            for i, (px, py) in enumerate(kpts):
                if px > 0 and py > 0: # Only draw valid detections
                    cv2.circle(frame, (int(px), int(py)), 5, (0, 255, 0), -1)
                    landmarks_dict[i] = (int(px), int(py))

            # --- SITTING DETECTION ---
            if 5 in landmarks_dict: # Left Shoulder
                shoulder_y = landmarks_dict[5][1]
                if self.baseline_shoulder_y is None:
                    self.baseline_shoulder_y = shoulder_y
                
                # If shoulders move up significantly, assume standing
                if shoulder_y < self.baseline_shoulder_y - 80:
                    is_sitting = False
                else:
                    is_sitting = True

            # --- DRINKING DETECTION ---
            # Check distance between Nose (0) and either Wrist (9 or 10)
            if 0 in landmarks_dict:
                nose_pos = landmarks_dict[0]
                for wrist_id in [9, 10]:
                    if wrist_id in landmarks_dict:
                        wrist_pos = landmarks_dict[wrist_id]
                        dist = math.hypot(wrist_pos[0] - nose_pos[0], wrist_pos[1] - nose_pos[1])
                        if dist < 70: # Pixel threshold for 'hand at face'
                            is_drinking = True

        return frame, is_sitting, is_drinking, landmarks_dict