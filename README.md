# DeskBot — Bottle/Person Pose & Hydration Detector

Short tagline: Real-time posture, gaze and bottle detection with hydration reminders and telemetry.

---

## Overview

DeskBot is a local, Python-based utility that uses YOLOv8 models to detect people, estimate pose landmarks, detect drinking behavior and water-vessel fill level, and provide ergonomic and hydration notifications. It runs locally with camera input, logs telemetry to a local SQLite DB, and issues desktop and voice alerts.

Target users: hobbyists, researchers, and developers building desktop ergonomic/hydration assistants or demos for computer-vision based user monitoring.

## Features

- Real-time human pose landmark extraction (Ultralytics YOLO pose model).
- Posture state detection (sitting vs away, slouch detection).
- Gaze-screen estimation (simple nose position heuristic).
- Drinking detection by proximity of wrist to nose.
- Bottle/cup detection with water-level estimation (pixel-edge analysis inside detected bounding box).
- Hydration reminders and low-water alerts via voice (pyttsx3 subprocess) and desktop notifications (plyer).
- Asynchronous AI inference worker to keep UI responsive.
- Local telemetry storage (`deskbot_metrics.db`) for session analytics.

## Tech Stack

- Language: Python 3.12+ (see `pyproject.toml`) — the code is compatible with Python 3.10+ in practice.
- Computer Vision: OpenCV (`opencv-python`), NumPy
- Models / Inference: Ultralytics YOLO (via the `ultralytics` package)
- Optional: ONNX/`onnxruntime` (ONNX model handling is referenced but not required by the core path)
- Data storage: SQLite (local file)
- Notifications: `plyer` (desktop) and `pyttsx3` (voice, run in isolated subprocess)

There is no frontend or REST API in this repository; the app is a single-process desktop/CLI utility that displays a live OpenCV window.

## Architecture & Data Flow

High level components:

- `main.py` — application entry point. Captures camera frames, spawns the asynchronous inference thread, merges overlays, draws dashboard, and manages high-level timers & alerts.
- `pose_detector.py` (`UnifiedEdgeDetector`) — loads an Ultralytics YOLO pose model, extracts landmarks, computes posture metrics, detects drinking gestures, and estimates water level inside detected vessel bounding boxes.
- `bottle_detector.py` (`BottleDetector`) — separate detector wrapper (note: references a config variable `YOLO_MODEL_PATH` which is not defined in `config.py`; see notes below).
- `hydration_manager.py` — manages hydration timing and fires hydration-related alerts.
- `notification_manager.py` — emits desktop notifications and manages a separate voice subprocess to run `pyttsx3` safely.
- `utils.py` — telemetry DB setup (`deskbot_metrics.db`) and dashboard drawing.

Data flow summary:

1. `main.py` captures frames from the camera and pushes a copy to the inference thread under a lock.
2. `UnifiedEdgeDetector.process_frame` runs the YOLOv8 pose/detection model, returns posture flags, vessel detection, water level estimates and overlay primitives.
3. `main.py` consumes the returned metrics, updates accumulators (sitting/standing, eye strain), calls `HydrationManager.update`, and triggers `NotificationManager.send_alert` when thresholds are met.
4. Telemetry is periodically appended to the local SQLite DB by `utils.append_telemetry`.

## Installation & Setup

Prerequisites

- Python 3.12+ recommended (the `pyproject.toml` specifies `requires-python = ">=3.12"`).
- A working camera or video source.

Clone the repo

```bash
git clone <repository-url>
cd pose
```

Create a virtual environment and install dependencies

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `requirements.txt` already lists `ultralytics`, `opencv-python`, `numpy`, `mediapipe`, `pyttsx3`, `plyer`, and `scipy`.
- If you require GPU acceleration, install a CUDA-compatible `torch` wheel before installing other dependencies per the PyTorch install guide.

Model files

Place model files in the project root. The project expects at least the pose model named in `config.py`:

- `yolov8n-pose.pt` — referenced by `config.YOLO_POSE_MODEL` and loaded by `pose_detector.py`.

`bottle_detector.py` references `config.YOLO_MODEL_PATH` for a detection model; this variable is not defined in `config.py` in the current code — if you plan to use `BottleDetector`, add `YOLO_MODEL_PATH = "yolov8n.pt"` (or another detection model path) to `config.py`.

## Project Structure

- [main.py](main.py) — entrypoint, UI loop, inference orchestration
- [config.py](config.py) — application configuration values and thresholds
- [pose_detector.py](pose_detector.py) — UnifiedEdgeDetector implementation (YOLO pose + vessel analysis)
- [bottle_detector.py](bottle_detector.py) — BottleDetector class (separate wrapper)
- [hydration_manager.py](hydration_manager.py) — hydration reminder logic
- [notification_manager.py](notification_manager.py) — desktop + voice notifications (multiprocessing)
- [utils.py](utils.py) — telemetry DB and dashboard renderer
- [requirements.txt](requirements.txt) — Python dependencies
- [pyproject.toml](pyproject.toml) — project metadata & declared dependencies
- model files: `yolov8n-pose.pt`, `yolov8n.pt` (if used), `yolov8n-pose.onnx` (optional)

## Configuration

Configuration is performed via `config.py`. Important settings include:

- `CAMERA_INDEX` — camera device index (default 0)
- `FRAME_WIDTH`, `FRAME_HEIGHT` — capture resolution
- `YOLO_POSE_MODEL` — path to the Ultralytics pose `.pt` file
- `BOTTLE_CLASS_ID`, `CUP_CLASS_ID` — expected class IDs for vessel detection
- `SITTING_TIME_THRESHOLD`, `STAND_RESET_THRESHOLD`, `EYE_STRAIN_THRESHOLD_SEC` — ergonomics thresholds
- `HYDRATION_REMINDER_INTERVAL`, `LOW_WATER_THRESHOLD_PERCENT`, `STANDARD_VESSEL_CAPACITY_ML` — hydration settings
- `ENABLE_VOICE_ALERTS`, `ENABLE_DESKTOP_ALERTS`, `NOTIFICATION_COOLDOWN` — notification controls

If you prefer to use environment variables, implement a small loader to populate `config.py` from `os.environ` (not currently implemented).

## Usage

Start the app (project root):

```powershell
python main.py
```

Behavior overview:

- On launch the system calibrates posture for a few seconds; it prints "Calibration Successful!" when done.
- The overlay shows detected landmarks, vessel bounding boxes, session timers and metrics.
- Alerts will be shown on-screen and spoken (if `ENABLE_VOICE_ALERTS=True`).

Debugging individual components:

```powershell
python pose_detector.py        # Runs the detector code path (may require small wrapper)
python bottle_detector.py      # Runs the bottle detector class logic (requires YOLO_MODEL_PATH in config)
```

## API / Endpoints

This repository does not expose HTTP APIs or an external service interface — it is a local application.

## Environment Variables

There are no required `.env` environment variables out of the box. The primary runtime configuration uses `config.py`. If you convert to `.env`, include equivalents for the `config.py` keys listed in the Configuration section.

## Troubleshooting

- Models not found: ensure `yolov8n-pose.pt` (and any other model paths) are present in the project root and match names referenced in `config.py`.
- `bottle_detector.py` fails to load: add `YOLO_MODEL_PATH = "yolov8n.pt"` to `config.py` or update the module to use `YOLO_POSE_MODEL`.
- Camera not accessible: check camera permissions, try different `CAMERA_INDEX` values, or supply a video file path in place of `cv2.VideoCapture(config.CAMERA_INDEX)`.
- pyttsx3 audio errors or device locks: the voice worker isolates `pyttsx3` into a subprocess to reduce lockups. If audio initialization fails, test `pyttsx3` separately in an interactive session.
- Torch / CUDA errors: install a `torch` build that matches your CUDA driver or use CPU-only builds for compatibility.

## Future Improvements

- Add CLI flags for overriding `config.py` values at runtime (model paths, device, source).
- Consolidate model-loading paths to avoid undefined config keys (`YOLO_MODEL_PATH`).
- Add automated unit tests for `hydration_manager.py` and detector wrappers.
- Create an optional headless mode to run on servers (no OpenCV GUI) and output telemetry to a remote store.
- Add Dockerfile and instructions for lightweight deployment.
- Implement an ONNX-first inference path using `onnxruntime` for platforms without PyTorch.

## License

No license file detected in repository. Suggested: add an open-source license such as MIT if you want to permit reuse.

---
