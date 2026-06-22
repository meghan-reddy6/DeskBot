# DeskBot Production Environment

The DeskBot 3-Gate Hybrid Engine is completely dockerized to run as an isolated, sub-100ms background daemon. This repository handles direct hardware passthroughs for the camera subsystem (`/dev/videoX`) and ALSA sound drivers (`/dev/snd`).

## 1. Directory Blueprint
Your deployment environment must mirror the following architecture before initiating the build sequence:

```text
/deskbot-deployment
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
├── config.py
├── pose_detector.py
├── utils.py
├── yolov8n-pose.pt      # Optional (PyTorch runtime fallback)
└── yolov8n-pose.onnx    # Required (Embedded edge optimization)
```

## 2. Host System Configuration (Linux / Embedded)
DeskBot requires strict host-level device access. If you are deploying on a native Linux or embedded host (like the Rubik Pi), you must append your runtime user to the hardware driver groups. Failure to do this will result in GStreamer and ALSA Docker passthrough permission failures.

Run the following commands on your host system:
```bash
# Grant access to video/camera streams
sudo usermod -aG video $USER

# Grant access to sound/ALSA devices
sudo usermod -aG audio $USER

# Apply the new group policies
newgrp video
newgrp audio
```

## 3. Daemon Execution Lifecycle
Execute the following container lifecycle commands from inside the target deployment folder.

**Step A: Build the Container Layer**
To prevent caching stale artifacts and ensure a clean environment lock, force an unpolluted compile:
```bash
docker-compose build --no-cache
```

**Step B: Background Detached Execution**
Spin up the orchestrator and drop it into a headless background daemon:
```bash
docker-compose up -d
```

**Step C: Telemetry & Log Monitoring**
Since the environment runs in `-d` (detached) mode, you can audit the core loop health and telemetry output via standard container logs:
```bash
docker logs -f deskbot_runtime
```

## 4. Teardown
To halt the daemon without destroying the persistent `sqlite` databases:
```bash
docker-compose down
```
