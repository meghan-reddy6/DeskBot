import cv2
import math
import numpy as np
from ultralytics import YOLO
import config

class UnifiedEdgeDetector:
    def __init__(self) -> None:
        self.model = YOLO(config.YOLO_POSE_MODEL)
        
        self.calibrated: bool = False
        self.baseline_neck_ratio: float = 0.0
        self.baseline_nose_y: float = 0.0
        
        self.neck_ratio_history: list[float] = []
        self.nose_y_history: list[float] = []
        self.water_level_history: list[float] = []

    def calculate_distance(self, pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
        """Calculates Euclidean distance between two coordinate tuples."""
        return float(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

    def estimate_fluid_volume(self, frame: np.ndarray, bbox: list[int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: 
            return 0.0, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        horizontal_sum = np.sum(edges, axis=1)
        total_height = horizontal_sum.shape[0]
        vertical_margin = int(total_height * 0.15)
        
        if total_height - vertical_margin <= vertical_margin: 
            return 50.0, config.STANDARD_VESSEL_CAPACITY_ML * 0.5
            
        valid_rows = horizontal_sum[vertical_margin:total_height - vertical_margin]
        if len(valid_rows) == 0 or np.max(valid_rows) == 0: 
            return 0.0, 0.0
            
        water_line_idx = np.argmax(valid_rows) + vertical_margin
        fill_percentage = min(max(((total_height - water_line_idx) / total_height) * 100, 0), 100)
        
        self.water_level_history.append(fill_percentage)
        if len(self.water_level_history) > config.SMOOTHING_WINDOW_SIZE:
            self.water_level_history.pop(0)
        smoothed_pct = float(np.mean(self.water_level_history))
        
        volume_ml = (smoothed_pct / 100.0) * config.STANDARD_VESSEL_CAPACITY_ML
        return smoothed_pct, volume_ml

    def process_frame(self, frame: np.ndarray):
        results = self.model(frame, verbose=False)[0]
        
        is_sitting = False
        is_slouching = False
        is_drinking = False
        is_gazing_screen = False
        vessel_detected = False
        vessel_type = "None"
        water_level_pct = None
        current_volume_ml = 0.0
        
        landmarks_dict = {}
        overlay_elements = []

        # --- ADVANCED BACKGROUND FILTER ENGINE ---
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            all_skeletons = results.keypoints.xy.cpu().numpy()
            all_confs = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else [None] * len(all_skeletons)
            
            best_person_idx = -1
            max_shoulder_width = -1.0
            target_kpts = None
            target_confs = None

            # Iterate through EVERY person in the camera view to find the closest one
            for p_idx, kpts in enumerate(all_skeletons):
                # Check Left Shoulder (5) and Right Shoulder (6) values
                if kpts[5][0] > 0 and kpts[5][1] > 0 and kpts[6][0] > 0 and kpts[6][1] > 0:
                    width = math.hypot(kpts[5][0] - kpts[6][0], kpts[5][1] - kpts[6][1])
                    # The person with the largest shoulder width is physically closest to the camera
                    if width > max_shoulder_width:
                        max_shoulder_width = width
                        best_person_idx = p_idx
                        target_kpts = kpts
                        if all_confs[p_idx] is not None:
                            target_confs = all_confs[p_idx]

            # If no shoulders were detected, default fallback to index 0
            if best_person_idx == -1:
                target_kpts = all_skeletons[0]
                if all_confs[0] is not None:
                    target_confs = all_confs[0]

            # Parse Landmarks for the anchored primary target only
            if target_kpts is not None:
                confs = target_confs if target_confs is not None else [1.0] * 17
                for idx, (x, y) in enumerate(target_kpts):
                    if x > 0 and y > 0 and confs[idx] > config.POSTURE_CONFIDENCE_THRESHOLD:
                        landmarks_dict[idx] = (int(x), int(y))
                        overlay_elements.append(("circle", (int(x), int(y)), 4, (0, 255, 0), -1))

            # --- SITTING & SLOUCH CALCULATIONS (RUN ON ANCHORED TARGET ONLY) ---
            if 0 in landmarks_dict and 5 in landmarks_dict and 6 in landmarks_dict:
                nose_y = float(landmarks_dict[0][1])
                shoulder_width = self.calculate_distance(landmarks_dict[5], landmarks_dict[6])
                shoulder_center_y = (landmarks_dict[5][1] + landmarks_dict[6][1]) / 2.0
                
                ear_y = float(landmarks_dict[3][1] if 3 in landmarks_dict else (landmarks_dict[4][1] if 4 in landmarks_dict else nose_y))
                vertical_neck_compression = abs(shoulder_center_y - ear_y)
                current_neck_ratio = vertical_neck_compression / max(shoulder_width, 1.0)
                
                self.neck_ratio_history.append(current_neck_ratio)
                self.nose_y_history.append(nose_y)
                
                if len(self.neck_ratio_history) > config.SMOOTHING_WINDOW_SIZE:
                    self.neck_ratio_history.pop(0)
                    self.nose_y_history.pop(0)
                
                smoothed_neck_ratio = float(np.mean(self.neck_ratio_history))
                smoothed_nose_y = float(np.mean(self.nose_y_history))

                if self.calibrated:
                    if smoothed_nose_y < (self.baseline_nose_y - (shoulder_width * 0.5)):
                        is_sitting = False
                    else:
                        is_sitting = True
                        
                    if smoothed_neck_ratio < (self.baseline_neck_ratio * 0.85):
                        is_slouching = True
                else:
                    is_sitting = True
                    is_slouching = False

            if 0 in landmarks_dict:
                nose_x = landmarks_dict[0][0]
                if abs(nose_x - (config.FRAME_WIDTH // 2)) < 120:
                    is_gazing_screen = True

            if 0 in landmarks_dict:
                nose_point = landmarks_dict[0]
                for wrist_idx in [9, 10]:
                    if wrist_idx in landmarks_dict:
                        if math.hypot(landmarks_dict[wrist_idx][0] - nose_point[0], landmarks_dict[wrist_idx][1] - nose_point[1]) < 75:
                            is_drinking = True

        # Parse Workspace Objects
        if results.boxes is not None:
            highest_confidence = 0.0
            selected_box = None
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id in [config.BOTTLE_CLASS_ID, config.CUP_CLASS_ID] and confidence > 0.45:
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        selected_box = [int(val) for val in box.xyxy[0]]
                        vessel_detected = True
                        vessel_type = "Water Bottle" if class_id == config.BOTTLE_CLASS_ID else "Cup/Mug"

            if vessel_detected and selected_box is not None:
                water_level_pct, current_volume_ml = self.estimate_fluid_volume(frame, selected_box)
                overlay_elements.append(("rect", (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), (255, 100, 0), 2))

        return (is_sitting, is_slouching, is_drinking, is_gazing_screen, 
                vessel_detected, vessel_type, water_level_pct, current_volume_ml, overlay_elements)