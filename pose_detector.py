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
        self.baseline_neck_ratio = 0.0
        self.baseline_nose_y = 0.0
        
        self.neck_ratio_history = []
        self.nose_y_history = []
        self.water_level_history = []
        
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
            # Anchor to the largest skeleton (closest person)
            best_idx = 0
            max_width = -1
            for idx, kpts in enumerate(keypoints_list):
                if kpts[5][2] > POSTURE_CONFIDENCE_THRESHOLD and kpts[6][2] > POSTURE_CONFIDENCE_THRESHOLD:
                    w = self.calculate_distance((kpts[5][0], kpts[5][1]), (kpts[6][0], kpts[6][1]))
                    if w > max_width:
                        max_width = w
                        best_idx = idx
                        
            target_kpts = keypoints_list[best_idx]
            
            # Dictionary of confident landmarks
            ldm = {}
            for i, (x, y, conf) in enumerate(target_kpts):
                if conf > POSTURE_CONFIDENCE_THRESHOLD:
                    ldm[i] = (int(x), int(y))
                    if not self.headless:
                        cv2.circle(annotated_frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # Sitting & Slouch Logic
            if 0 in ldm and 5 in ldm and 6 in ldm:
                nose_y = float(ldm[0][1])
                shoulder_width = self.calculate_distance(ldm[5], ldm[6])
                shoulder_center_y = (ldm[5][1] + ldm[6][1]) / 2.0
                
                ear_y = float(ldm[3][1] if 3 in ldm else (ldm[4][1] if 4 in ldm else nose_y))
                vertical_neck_compression = abs(shoulder_center_y - ear_y)
                current_neck_ratio = vertical_neck_compression / max(shoulder_width, 1.0)
                
                self.neck_ratio_history.append(current_neck_ratio)
                self.nose_y_history.append(nose_y)
                if len(self.neck_ratio_history) > SMOOTHING_WINDOW_SIZE:
                    self.neck_ratio_history.pop(0)
                    self.nose_y_history.pop(0)
                
                smoothed_neck_ratio = float(np.mean(self.neck_ratio_history))
                smoothed_nose_y = float(np.mean(self.nose_y_history))

                if self.calibrated:
                    if smoothed_nose_y < (self.baseline_nose_y - (shoulder_width * 0.5)):
                        metrics["is_sitting"] = False
                    else:
                        metrics["is_sitting"] = True
                        if smoothed_neck_ratio < (self.baseline_neck_ratio * 0.85):
                            metrics["is_slouching"] = True
                else:
                    metrics["is_sitting"] = True # Default while calibrating

            # Gaze Logic
            if 0 in ldm:
                nose_x = ldm[0][0]
                if abs(nose_x - (FRAME_WIDTH // 2)) < 120:
                    metrics["is_gazing_screen"] = True

            # Drinking Logic
            if 0 in ldm:
                for wrist in [9, 10]:
                    if wrist in ldm:
                        if self.calculate_distance(ldm[0], ldm[wrist]) < 75:
                            metrics["is_drinking"] = True

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
# Mock Execution Wrapper
# ==========================================
if __name__ == "__main__":
    print("Initializing mock execution of UnifiedEdgeDetector...")
    
    # Create a dummy noisy image frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test 1: PyTorch Backend
    try:
        print("\n--- Testing PYTORCH Backend ---")
        import os
        # Fallback to yolov8n.pt if pose model isn't available for mock tests
        pt_model = "yolov8n-pose.pt" if os.path.exists("yolov8n-pose.pt") else "yolov8n.pt"
        
        detector_pt = UnifiedEdgeDetector(backend="PYTORCH", model_path=pt_model, headless=True)
        start = time.time()
        metrics_pt, _ = detector_pt.process_frame(dummy_frame)
        duration_pt = time.time() - start
        print(f"Metrics Output: {metrics_pt}")
        print(f"Inference Time: {duration_pt:.4f}s")
    except Exception as e:
        print(f"PyTorch backend test failed: {e}")
        
    # Test 2: ONNX Backend
    try:
        print("\n--- Testing ONNX Backend ---")
        onnx_model = "yolov8n-pose.onnx"
        if os.path.exists(onnx_model):
            # Install onnxruntime via: pip install onnxruntime
            detector_onnx = UnifiedEdgeDetector(backend="ONNX", model_path=onnx_model, headless=True)
            start = time.time()
            metrics_onnx, _ = detector_onnx.process_frame(dummy_frame)
            duration_onnx = time.time() - start
            print(f"Metrics Output: {metrics_onnx}")
            print(f"Inference Time: {duration_onnx:.4f}s")
        else:
            print(f"Skipping ONNX test: '{onnx_model}' not found locally.")
    except Exception as e:
        print(f"ONNX backend test failed: {e}")