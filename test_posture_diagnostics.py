import cv2
import time
import sys
import numpy as np
from pose_detector import UnifiedEdgeDetector
import config

class DiagnosticDetector(UnifiedEdgeDetector):
    def __init__(self):
        super().__init__()
        self.diag_counter = 0

    def process_frame(self, frame):
        metrics = {
            "is_sitting": True,
            "is_slouching": False,
            "vessel_type": None,
            "water_level_pct": 0,
            "current_volume_ml": 0
        }
        annotated_frame = frame.copy()
        self.diag_counter += 1

        boxes, class_ids, confs, keypoints_list = self._run_inference(frame)
        
        person_boxes = []
        person_keypoints = []
        
        # Filter for persons
        for i, class_id in enumerate(class_ids):
            if int(class_id) == 0:
                person_boxes.append(boxes[i])
                if i < len(keypoints_list):
                    person_keypoints.append(keypoints_list[i])

        if person_boxes and person_keypoints:
            best_idx = 0
            max_area = 0
            for i, box in enumerate(person_boxes):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    best_idx = i
                    
            target_kpts = person_keypoints[best_idx]
            target_box = person_boxes[best_idx]

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

            shoulders_visible = (conf_L_shoulder > config.POSE_SHOULDER_MIN_CONF and 
                                 conf_R_shoulder > config.POSE_SHOULDER_MIN_CONF)
            nose_lost = (conf_nose < config.POSE_NOSE_LOST_CONF)

            shoulder_center_y = 0.0
            if shoulders_visible:
                shoulder_center_y = (target_kpts[5][1] + target_kpts[6][1]) / 2.0

            # Rule States for printing
            rule_vertical_span = False
            rule_modeA_torso = False
            rule_modeB_aspect = False
            rule_modeB_centroid = False

            if current_height > (frame.shape[0] * 0.85):
                raw_metrics["is_sitting"] = False
                rule_vertical_span = True
            else:
                if shoulders_visible:
                    current_Sb = self.calculate_distance(target_kpts[5], target_kpts[6])
                    
                    if not self.calibrated:
                        if not nose_lost:
                            self.calib_Sb.append(current_Sb)
                            self.calib_aspect_ratio.append(current_aspect_ratio)
                            self.calib_centroid_y.append(current_centroid_y)
                            
                            nose_y = target_kpts[0][1]
                            torso_ratio = abs(shoulder_center_y - nose_y) / max(current_Sb, 1.0)
                            self.calib_torso_ratio.append(torso_ratio)
                            self.calib_nose_to_box.append(abs(nose_y - by1) / max(current_height, 1.0))
                    else:
                        if not nose_lost:
                            # Mode A
                            nose_y = target_kpts[0][1]
                            current_torso_ratio = abs(shoulder_center_y - nose_y) / max(current_Sb, 1.0)
                            
                            if current_torso_ratio > (self.base_torso_ratio * config.STANDING_EXTENSION_SENSITIVITY):
                                raw_metrics["is_sitting"] = False
                                rule_modeA_torso = True
                                
                            current_nose_to_box_ratio = abs(nose_y - by1) / max(current_height, 1.0)
                            if raw_metrics["is_sitting"] and current_nose_to_box_ratio > (self.base_nose_to_box * config.SLOUCH_COMPRESSION_SENSITIVITY):
                                raw_metrics["is_slouching"] = True
                        else:
                            # Mode B
                            upward_shift = self.base_centroid_y - current_centroid_y
                            if current_aspect_ratio > (self.base_aspect_ratio * 1.30):
                                raw_metrics["is_sitting"] = False
                                rule_modeB_aspect = True
                            elif upward_shift > (self.base_Sb * 0.40):
                                raw_metrics["is_sitting"] = False
                                rule_modeB_centroid = True
                else:
                    if self.calibrated:
                        upward_shift = self.base_centroid_y - current_centroid_y
                        if current_aspect_ratio > (self.base_aspect_ratio * 1.30):
                            raw_metrics["is_sitting"] = False
                            rule_modeB_aspect = True
                        elif upward_shift > (self.base_Sb * 0.40):
                            raw_metrics["is_sitting"] = False
                            rule_modeB_centroid = True

            # Absolute Sitting Reclamation Rule
            if shoulders_visible and not nose_lost and shoulder_center_y > (frame.shape[0] * 0.40):
                raw_metrics["is_sitting"] = True

            if self.diag_counter % 15 == 0:
                print("\n" + "="*60)
                print(">>> RAW TELEMETRY REPORT <<<")
                print("-" * 60)
                print(f"[Bounding Box] Height: {current_height:.1f}px | Width: {current_width:.1f}px | Aspect: {current_aspect_ratio:.3f} | Centroid Y: {current_centroid_y:.1f}")
                print(f"[Keypoints] Nose Conf: {conf_nose:.2f} | L-Shld Conf: {conf_L_shoulder:.2f} | R-Shld Conf: {conf_R_shoulder:.2f}")
                
                if shoulders_visible:
                    print(f"            Shoulder Center Y: {shoulder_center_y:.1f}")
                    
                if self.calibrated:
                    print("-" * 60)
                    print(f"[Active Calibration Vectors]")
                    print(f"  base_Sb:           {self.base_Sb:.1f}")
                    print(f"  base_aspect_ratio: {self.base_aspect_ratio:.3f}")
                    print(f"  base_centroid_y:   {self.base_centroid_y:.1f}")
                    print(f"  base_torso_ratio:  {self.base_torso_ratio:.3f}")
                    print("-" * 60)
                    print(f"[Evaluated Conditions]")
                    print(f"  Vertical Span Rule (>85%):    {rule_vertical_span}")
                    print(f"  Mode A Torso Ratio Trigger:   {rule_modeA_torso}")
                    print(f"  Mode B Aspect Expansion:      {rule_modeB_aspect}")
                    print(f"  Mode B Centroid Shift:        {rule_modeB_centroid}")
                    print(f"  Sitting Reclamation Override: {shoulders_visible and not nose_lost and shoulder_center_y > (frame.shape[0] * 0.40)}")
                    print(f"  Final Raw Prediction:         {raw_metrics['is_sitting']}")

        else:
            raw_metrics = {
                "is_sitting": False,
                "is_slouching": False,
                "is_gazing_screen": False,
                "is_drinking": False
            }

        # Saturated Hysteresis Buffer
        for state, raw_flag in raw_metrics.items():
            buf = self.state_buffers[state]
            buf.append(raw_flag)
            if len(buf) > 3:
                buf.pop(0)
                
            if all(val == True for val in buf):
                self.current_states[state] = True
            elif all(val == False for val in buf):
                self.current_states[state] = False
                
        if self.diag_counter % 15 == 0:
            print("-" * 60)
            print(f"[Hysteresis Buffer] 'is_sitting' ring buffer: {self.state_buffers['is_sitting']}")
            print(f"[Public UI State] Sitting: {self.current_states.get('is_sitting', True)}")
            print("="*60)

        metrics["is_sitting"] = self.current_states.get("is_sitting", True)
        return metrics, annotated_frame


def run_diagnostics():
    print("\n============================================================")
    print("DESKBOT STANDALONE POSTURE DIAGNOSTICS MODULE")
    print("============================================================")
    print("This module will isolate the camera and execute YOLO inference.")
    print("Please follow the instructions on screen.\n")
    
    time.sleep(2)
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    detector = DiagnosticDetector()
    
    print("\n[PHASE 1] Initializing hardware & starting 4-second calibration...")
    print("-> Position 1: Sit normally in your workspace chair. STAY STILL.")
    
    calibration_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        metrics, display_frame = detector.process_frame(frame)
        
        cv2.putText(display_frame, "DIAGNOSTIC MODE ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Diagnostic Feed", display_frame)
        
        if not detector.calibrated:
            if time.time() - calibration_start > 4.0:
                if len(detector.calib_aspect_ratio) > 10:
                    detector.base_aspect_ratio = sum(detector.calib_aspect_ratio) / len(detector.calib_aspect_ratio)
                    detector.base_Sb = sum(detector.calib_Sb) / len(detector.calib_Sb)
                    detector.base_centroid_y = sum(detector.calib_centroid_y) / len(detector.calib_centroid_y)
                    detector.base_torso_ratio = sum(detector.calib_torso_ratio) / len(detector.calib_torso_ratio)
                    detector.base_nose_to_box = sum(detector.calib_nose_to_box) / len(detector.calib_nose_to_box)
                else:
                    detector.base_aspect_ratio = 1.2
                    detector.base_Sb = 150.0
                    detector.base_centroid_y = config.FRAME_HEIGHT / 2.0
                    detector.base_torso_ratio = 0.75
                    detector.base_nose_to_box = 50.0
                detector.calibrated = True
                print("\n[CALIBRATION LOCKED] You may now move.")
                print("-> Position 2: Stand up close to the camera lens (clip your head/shoulders).")
                print("-> Position 3: Sit back down in your normal chair.")
                print("\nWatch the terminal output. Press 'q' in the video window to quit.")
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostics()
