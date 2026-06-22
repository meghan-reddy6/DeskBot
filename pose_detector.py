import cv2
import math
import numpy as np
import time
import importlib

# Attempt to import config for defaults, but allow fallback if missing
try:
    import config
    BOTTLE_CLASS_ID = config.BOTTLE_CLASS_ID
    CUP_CLASS_ID = config.CUP_CLASS_ID
    FRAME_WIDTH = config.FRAME_WIDTH
    STANDARD_VESSEL_CAPACITY_ML = config.STANDARD_VESSEL_CAPACITY_ML
    SMOOTHING_WINDOW_SIZE = config.SMOOTHING_WINDOW_SIZE
    POSTURE_CONFIDENCE_THRESHOLD = config.POSTURE_CONFIDENCE_THRESHOLD
except ImportError:
    BOTTLE_CLASS_ID = 39
    CUP_CLASS_ID = 41
    FRAME_WIDTH = 640
    STANDARD_VESSEL_CAPACITY_ML = 500.0
    SMOOTHING_WINDOW_SIZE = 5
    POSTURE_CONFIDENCE_THRESHOLD = 0.40

class UnifiedEdgeDetector:
    def __init__(self, backend="PYTORCH", model_path=None, headless=False):
        """
        Unified detector capable of switching between PyTorch (dev) and ONNX (production/edge)
        without altering the downstream heuristic pipeline.
        """
        self.backend = backend.upper()
        self.headless = headless
        self.model_path = model_path
        
        # State tracking and calibration
        self.calibrated = False
        
        # Dual-Mode Adaptive Calibration Tracking Arrays
        self.calib_Sb = []
        self.calib_aspect_ratio = []
        self.calib_centroid_y = []
        self.calib_torso_ratio = []
        self.calib_nose_to_box = []
        
        # Stored Structural Constants (locked after calibration)
        self.base_Sb = 150.0 
        self.base_aspect_ratio = 1.2
        self.base_centroid_y = 0.0
        self.base_torso_ratio = 0.75
        self.base_nose_to_box = 0.0
        
        self.water_level_history = []
        self.locked_target_pos = None  # (cx, cy) tuple for tracking user

        self.frame_counter = 0
        
        # Instant Saturated Hysteresis (3-frame buffer)
        self.state_buffers = {
            "is_sitting": [True, True, True],
            "is_slouching": [False, False, False],
            "is_drinking": [False, False, False],
            "is_gazing_screen": [False, False, False]
        }
        self.current_states = {
            "is_sitting": True,
            "is_slouching": False,
            "is_drinking": False,
            "is_gazing_screen": False
        }

        self.model = None
        self.ort_session = None
        self.input_name = None
        
        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "PYTORCH":
            if self.model_path is None:
                self.model_path = "yolov8n-pose.pt"
            # Dynamically import ultralytics to avoid memory bloat when using ONNX
            ultralytics = importlib.import_module('ultralytics')
            self.model = ultralytics.YOLO(self.model_path)
            
        elif self.backend == "ONNX":
            if self.model_path is None:
                self.model_path = "yolov8n-pose.onnx"
            # Dynamically import onnxruntime
            ort = importlib.import_module('onnxruntime')
            # Initialize ORT session (CPU or accelerated provider if available)
            self.ort_session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name
            
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _preprocess_onnx(self, frame):
        """Prepare frame for ONNX inference."""
        # Resize to 640x640 (standard YOLOv8 input size)
        img = cv2.resize(frame, (640, 640))
        # Convert BGR to RGB and transpose to CHW
        img = img.transpose((2, 0, 1))[::-1] 
        img = np.ascontiguousarray(img)
        # Normalize to 0.0 - 1.0
        img = img.astype(np.float32)
        img /= 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
        
    def _postprocess_onnx(self, preds, orig_shape):
        """
        Parses YOLOv8-pose ONNX output.
        Output shape is typically [1, 56, 8400] for pose models.
        56 features = 4 (bbox) + 1 (object conf/class) + 51 (17 keypoints * 3 (x,y,conf))
        We parse this into a unified list structure.
        """
        preds = preds[0] # remove batch dim -> [56, 8400]
        preds = np.transpose(preds) # -> [8400, 56]
        
        boxes = []
        keypoints_list = []
        confs = []
        class_ids = []
        
        orig_h, orig_w = orig_shape[:2]
        x_factor = orig_w / 640.0
        y_factor = orig_h / 640.0
        
        # Parse detections (Simplified confidence thresholding for edge implementation)
        for row in preds:
            box_conf = row[4] 
            if box_conf > 0.5:
                cx, cy, w, h = row[0:4]
                x1 = int((cx - w/2) * x_factor)
                y1 = int((cy - h/2) * y_factor)
                x2 = int((cx + w/2) * x_factor)
                y2 = int((cy + h/2) * y_factor)
                
                boxes.append([x1, y1, x2, y2])
                confs.append(box_conf)
                class_ids.append(0) # Standard YOLOv8 pose only detects person (0)
                
                # Parse 17 keypoints (starting at index 5)
                kpts = []
                for i in range(17):
                    kx = row[5 + (i*3)] * x_factor
                    ky = row[5 + (i*3) + 1] * y_factor
                    kc = row[5 + (i*3) + 2]
                    kpts.append((kx, ky, kc))
                keypoints_list.append(kpts)
                
        # For a full production system, cv2.dnn.NMSBoxes would process the arrays here.
        # Returning best raw detection to feed heuristics.
        if len(boxes) > 0:
            best_idx = np.argmax(confs)
            return [boxes[best_idx]], [class_ids[best_idx]], [confs[best_idx]], [keypoints_list[best_idx]]
            
        return [], [], [], []

    def _run_inference(self, frame):
        """Returns uniform structure: boxes, class_ids, confs, keypoints_list"""
        if self.backend == "PYTORCH":
            results = self.model(frame, verbose=False)[0]
            boxes = []
            class_ids = []
            confs = []
            keypoints_list = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    boxes.append([int(v) for v in box.xyxy[0]])
                    class_ids.append(int(box.cls[0]))
                    confs.append(float(box.conf[0]))
                    
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                kpts_xy = results.keypoints.xy.cpu().numpy()
                kpts_conf = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None
                
                for i, person_kpts in enumerate(kpts_xy):
                    k_list = []
                    for j, (x, y) in enumerate(person_kpts):
                        c = kpts_conf[i][j] if kpts_conf is not None else 1.0
                        k_list.append((x, y, c))
                    keypoints_list.append(k_list)
                    
            return boxes, class_ids, confs, keypoints_list
            
        elif self.backend == "ONNX":
            img = self._preprocess_onnx(frame)
            preds = self.ort_session.run(None, {self.input_name: img})[0]
            return self._postprocess_onnx(preds, frame.shape)

    def calculate_distance(self, pt1, pt2):
        return float(math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))

    def estimate_fluid_volume(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
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
            return 50.0, STANDARD_VESSEL_CAPACITY_ML * 0.5
            
        valid_rows = horizontal_sum[vertical_margin:total_height - vertical_margin]
        if len(valid_rows) == 0 or np.max(valid_rows) == 0: 
            return 0.0, 0.0
            
        water_line_idx = np.argmax(valid_rows) + vertical_margin
        fill_percentage = min(max(((total_height - water_line_idx) / total_height) * 100, 0), 100)
        
        self.water_level_history.append(fill_percentage)
        if len(self.water_level_history) > SMOOTHING_WINDOW_SIZE:
            self.water_level_history.pop(0)
        smoothed_pct = float(np.mean(self.water_level_history))
        
        volume_ml = (smoothed_pct / 100.0) * STANDARD_VESSEL_CAPACITY_ML
        return smoothed_pct, volume_ml

    def process_frame(self, frame):
        """
        Executes unified inference and core DeskBot heuristics.
        Returns a structured dictionary of metrics, and optionally the annotated frame if not headless.
        """
        self.frame_counter += 1
        annotated_frame = frame.copy() if not self.headless else None
        boxes, class_ids, confs, keypoints_list = self._run_inference(frame)
        
        # Initialize cleanly structured metric payload
        metrics = {
            "is_sitting": False,
            "is_slouching": False,
            "is_drinking": False,
            "is_gazing_screen": False,
            "vessel_detected": False,
            "vessel_type": "None",
            "water_level_pct": None,
            "current_volume_ml": 0.0
        }
        
        # --- PERSON & POSTURE HEURISTICS ---
        if keypoints_list:
            best_idx = 0
            
            # Compute centers for all skeletons (use average of shoulders)
            centers = []
            valid_indices = []
            for idx, kpts in enumerate(keypoints_list):
                if kpts[5][2] > POSTURE_CONFIDENCE_THRESHOLD and kpts[6][2] > POSTURE_CONFIDENCE_THRESHOLD:
                    cx = (kpts[5][0] + kpts[6][0]) / 2.0
                    cy = (kpts[5][1] + kpts[6][1]) / 2.0
                    centers.append((cx, cy))
                    valid_indices.append(idx)
                    
            if valid_indices:
                if not self.calibrated or self.locked_target_pos is None:
                    # Calibration Phase: Lock to the center-most person
                    # Extract height and width dynamically from the incoming frame numpy array
                    height, width = frame.shape[:2]
                    frame_center = (width / 2.0, height / 2.0)
                    min_dist = float('inf')
                    for i, (cx, cy) in enumerate(centers):
                        dist = self.calculate_distance((cx, cy), frame_center)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = valid_indices[i]
                    # Set initial lock coordinates
                    self.locked_target_pos = centers[valid_indices.index(best_idx)]
                else:
                    # Proximity tracking to locked target (ignore background people)
                    min_dist = float('inf')
                    for i, (cx, cy) in enumerate(centers):
                        dist = self.calculate_distance((cx, cy), self.locked_target_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = valid_indices[i]
                            
                    # Update locked target position with momentum to prevent jitter
                    best_cx, best_cy = centers[valid_indices.index(best_idx)]
                    self.locked_target_pos = (
                        self.locked_target_pos[0] * 0.8 + best_cx * 0.2,
                        self.locked_target_pos[1] * 0.8 + best_cy * 0.2
                    )
            elif len(keypoints_list) > 0:
                # Fallback if no shoulders are highly confident
                best_idx = 0
                    
            target_kpts = keypoints_list[best_idx]
            target_box = boxes[best_idx]  # Extract corresponding bounding box
            
            # Dictionary of confident landmarks
            ldm = {}
            for i, (x, y, conf) in enumerate(target_kpts):
                if conf > POSTURE_CONFIDENCE_THRESHOLD:
                    ldm[i] = (int(x), int(y))
                    if not self.headless:
                        cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Bounding Box Geometry Calculations
            bx1, by1, bx2, by2 = target_box
            current_width = float(abs(bx2 - bx1))
            current_height = float(abs(by2 - by1))
            current_aspect_ratio = current_height / max(current_width, 1.0)
            current_centroid_y = by1 + (current_height / 2.0)

            raw_metrics = {
                "is_sitting": True,
                "is_slouching": False,
                "is_gazing_screen": True,
                "is_drinking": False
            }

            conf_L_shoulder = target_kpts[5][2]
            conf_R_shoulder = target_kpts[6][2]
            conf_nose = target_kpts[0][2]

            shoulders_visible = (conf_L_shoulder > 0.45 and conf_R_shoulder > 0.45)
            nose_lost = (conf_nose < 0.30)

            shoulder_center_y = 0.0
            if shoulders_visible:
                shoulder_center_y = (target_kpts[5][1] + target_kpts[6][1]) / 2.0
                current_Sb = self.calculate_distance(target_kpts[5], target_kpts[6])

            # 1. GLOBAL VERTICAL OVERRIDE (ZERO FAILURE)
            if current_height > (frame.shape[0] * 0.82):
                raw_metrics["is_sitting"] = False
            else:
                if not self.calibrated:
                    if shoulders_visible and not nose_lost:
                        self.calib_Sb.append(current_Sb)
                        self.calib_aspect_ratio.append(current_aspect_ratio)
                        self.calib_centroid_y.append(current_centroid_y)
                        
                        nose_y = target_kpts[0][1]
                        torso_ratio = abs(shoulder_center_y - nose_y) / max(current_Sb, 1.0)
                        self.calib_torso_ratio.append(torso_ratio)
                        self.calib_nose_to_box.append(abs(nose_y - by1) / max(current_height, 1.0))
                else:
                    # 2. HYBRID CLASSIFICATION ENGINE
                    if shoulders_visible and not nose_lost:
                        # Mode A (Standard Distance)
                        nose_y = target_kpts[0][1]
                        current_torso_ratio = abs(shoulder_center_y - nose_y) / max(current_Sb, 1.0)
                        
                        if current_torso_ratio > (self.base_torso_ratio * 1.15):
                            raw_metrics["is_sitting"] = False
                            
                        # 4. ACCURATE SLOUCH CONTROLLER
                        current_nose_to_box_ratio = abs(nose_y - by1) / max(current_height, 1.0)
                        if raw_metrics["is_sitting"] and current_nose_to_box_ratio > (self.base_nose_to_box * 1.15):
                            raw_metrics["is_slouching"] = True
                    else:
                        # Mode B (Clipping/Close-up Fallback)
                        upward_shift = self.base_centroid_y - current_centroid_y
                        if current_aspect_ratio > (self.base_aspect_ratio * 1.30):
                            raw_metrics["is_sitting"] = False
                        elif upward_shift > (frame.shape[0] * 0.25):
                            raw_metrics["is_sitting"] = False

            # Trigonometric Face Symmetry Evaluation (Gaze Logic)
            if 3 in ldm and 4 in ldm and 0 in ldm:
                dist_left = self.calculate_distance(ldm[0], ldm[3])
                dist_right = self.calculate_distance(ldm[0], ldm[4])
                ratio = dist_left / max(dist_right, 0.001)
                raw_metrics["is_gazing_screen"] = (0.60 < ratio < 1.66)
            else:
                raw_metrics["is_gazing_screen"] = True 
        else:
            # If no person bounding box is detected at all
            raw_metrics = {
                "is_sitting": False, # Instant Away/Standing
                "is_slouching": False,
                "is_gazing_screen": False,
                "is_drinking": False
            }

        # --- VESSEL HEURISTICS ---
        best_vessel_box = None
        best_vessel_conf = 0.0
        vessel_cls = None
        
        for i, box in enumerate(boxes):
            c_id = class_ids[i]
            conf = confs[i]
            if c_id in [BOTTLE_CLASS_ID, CUP_CLASS_ID] and conf > 0.45:
                if conf > best_vessel_conf:
                    best_vessel_conf = conf
                    best_vessel_box = box
                    vessel_cls = "Water Bottle" if c_id == BOTTLE_CLASS_ID else "Cup/Mug"

        if keypoints_list:
            # 4. Vessel-Bounded Drinking Guard
            if 0 in ldm and best_vessel_box:
                vessel_cx = (best_vessel_box[0] + best_vessel_box[2]) / 2.0
                vessel_cy = (best_vessel_box[1] + best_vessel_box[3]) / 2.0
                for wrist in [9, 10]:
                    if wrist in ldm:
                        dist_nose_wrist = self.calculate_distance(ldm[0], ldm[wrist])
                        dist_wrist_vessel = self.calculate_distance(ldm[wrist], (vessel_cx, vessel_cy))
                        
                        if dist_nose_wrist < (self.base_Sb * 0.45) and dist_wrist_vessel < (self.base_Sb * 0.5):
                            raw_metrics["is_drinking"] = True
                            break

            # 3. ABSOLUTE SITTING RECLAMATION
            if (conf_nose > 0.60 and conf_L_shoulder > 0.60 and conf_R_shoulder > 0.60):
                if shoulder_center_y > (frame.shape[0] * 0.45):
                    raw_metrics["is_sitting"] = True

        # 3. Instant Saturated Hysteresis (3-frame buffer)
        for state, raw_flag in raw_metrics.items():
            buf = self.state_buffers[state]
            buf.append(raw_flag)
            if len(buf) > 3:
                buf.pop(0)
                
            if all(val == True for val in buf):
                self.current_states[state] = True
            elif all(val == False for val in buf):
                self.current_states[state] = False
                
        # Transfer to final metrics payload
        metrics["is_sitting"] = self.current_states["is_sitting"]
        metrics["is_slouching"] = self.current_states["is_slouching"]
        metrics["is_drinking"] = self.current_states["is_drinking"]
        metrics["is_gazing_screen"] = self.current_states["is_gazing_screen"]

        if best_vessel_box:
            metrics["vessel_detected"] = True
            metrics["vessel_type"] = vessel_cls
            pct, ml = self.estimate_fluid_volume(frame, best_vessel_box)
            metrics["water_level_pct"] = pct
            metrics["current_volume_ml"] = ml
            
            if not self.headless:
                x1, y1, x2, y2 = best_vessel_box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                
        return metrics, annotated_frame

# ==========================================
# Diagnostic Evaluation Wrapper
# ==========================================
if __name__ == "__main__":
    import cv2
    import time
    print("=" * 60)
    print("POSE DETECTOR DIAGNOSTIC MODE")
    print("=" * 60)
    print("STEP 1: Sit normally and look at your terminal. Note down your 'Torso Ratio' value.")
    print("STEP 2: Stand up normally (or close up). Note down your 'Torso Ratio' and 'Shoulder Center Y' values.")
    print("-" * 60)
    print("HOW TO TUNE YOUR CONFIG.PY:")
    print("-> If you are standing but Torso Ratio < 1.05, you need to lower TORSO_STRAIGHT_THRESHOLD below your standing Torso Ratio.")
    print("-> If your head clips off screen when standing, ensure your 'Shoulder Center Y' is LESS THAN 'Frame Height Limit'. If not, increase TOP_FRAME_CLIP_BOUNDARY (e.g., to 0.40).")
    print("=" * 60)
    print("Starting live camera feed in 3 seconds...")
    time.sleep(3)

    detector = UnifiedEdgeDetector(backend="PYTORCH", headless=False)
    # Simulate calibration baseline so math matches production constraints
    detector.calibrated_Sb = 150.0 
    detector.calibrated = True # Force into active calculation state
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics, annotated = detector.process_frame(frame)
        if annotated is not None:
            cv2.imshow("Diagnostic Feed", annotated)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()