# Bottle/Person Pose & Hydration Detector

Version: 2.0

Changelog (v2.0):

- Updated `README.md` with clearer setup, run, and troubleshooting steps.
- Clarified available modules and files in the repository.
- Documented usage for `main.py`, `pose_detector.py`, and `bottle_detector.py`.

Small project for detecting people, bottles, and poses using YOLOv8 and utilities in this repository.

## Contents

- **Overview:** what this repo does
- **Requirements:** Python and packages
- **Setup:** create/activate virtualenv and install deps
- **Run:** commands to start the app
- **Configuration:** key files to edit
- **Troubleshooting:** common issues and fixes

## Overview

This project uses YOLOv8 models to detect bottles and human poses and runs utilities in the repository ([main.py](main.py), [pose_detector.py](pose_detector.py), [bottle_detector.py](bottle_detector.py), [hydration_manager.py](hydration_manager.py), [notification_manager.py](notification_manager.py)). Put model files in the project root and run `main.py` to start detection.

## Requirements

- Windows (tested)
- Python 3.10+ (3.12 compatible)
- A working webcam or video source
- The file [requirements.txt](requirements.txt) lists Python dependencies. Install them into a dedicated virtual environment.
- Optional: GPU acceleration (install a CUDA/CuDNN-compatible `torch` build).

Files of note:

- [main.py](main.py)
- [pose_detector.py](pose_detector.py)
- [bottle_detector.py](bottle_detector.py)
- [requirements.txt](requirements.txt)
- `yolov8n-pose.pt` and `yolov8n.pt` (model weights) — keep them in the project root.

## Setup (recommended)

1. Install Python 3.10+ from python.org if you don't have it.
2. (Optional) Use the included virtual environment located at `dbot/` or create a new one.

PowerShell (activate included `dbot` venv):

```powershell
.\dbot\Scripts\Activate.ps1
# or: .\dbot\Scripts\activate
```

Command Prompt (activate):

```cmd
dbot\Scripts\activate.bat
```

If you prefer a new venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If you want GPU acceleration, install an appropriate `torch` wheel for your CUDA version before or after the above step. See the `torch` installation instructions on the PyTorch website for the correct command for your GPU/OS.

## Model files

Place these model files in the project root (they are already present if you copied them here):

- `yolov8n-pose.pt` — pose model
- `yolov8n.pt` — detection/classification model

If you do not have them, download the correct YOLOv8 weights and name them accordingly.

## Run the app

Basic run (from project root):

```powershell
python main.py
```

If you need to run a specific module for debugging, you can call it directly, e.g.:

```powershell
python pose_detector.py
python bottle_detector.py
```

## Configuration

- [config.py](config.py) holds configurable parameters (camera index, thresholds, file paths). Edit it to tune behavior.
- [main.py](main.py) is the primary entry point and wires detectors, managers, and notifications together.

## Troubleshooting

- Missing packages after `pip install -r requirements.txt`: ensure your venv is activated and you used the correct Python interpreter.
- Model files not found: confirm `yolov8n-pose.pt` and `yolov8n.pt` exist in the project root.
- Camera not detected: check Windows privacy settings and camera index in [config.py](config.py). Try different camera indices (0, 1, 2).
- GPU/torch errors: install the correct `torch` for your CUDA version; if unsure, try CPU-only `pip install torch --index-url https://download.pytorch.org/whl/cpu` or rely on the `requirements.txt` defaults.
- Permission errors when opening camera or saving files: run the terminal as Administrator or adjust file paths in [config.py](config.py) to a writable folder.

If you encounter specific tracebacks, copy the error into an issue or contact the maintainer (see below).

## Tests and Development

- This repository does not include automated tests. For local development, run individual modules and validate behavior from logs.

## Next steps / Suggestions

- Add unit tests for detection and manager logic.
- Add CLI flags to `main.py` for selecting models, camera, or running a dry-run.
- Add a dockerfile or cross-platform instructions if you want non-Windows support.

## License & Contact

This README is provided as-is. For questions or to contribute, open an issue or contact the repository owner.
