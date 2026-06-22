# DeskBot Edge Deployment Runbook

This runbook provides step-by-step technical instructions for deploying the DeskBot application securely via Docker on both local development environments and hardware-accelerated edge targets like the Qualcomm Rubik Pi.

---

## 1. System Prerequisites & Permissions

### Host-Level Camera Permissions (Rubik Pi / Linux)
To allow the Docker container to access hardware cameras (via standard V4L2 or GStreamer's `qtiqmmfsrc`), the host's `docker` daemon and the container user must have permissions to read `/dev/video*` devices natively.

1. **Add your host user to the video group:**
   ```bash
   sudo usermod -aG video $USER
   ```
2. **Explicitly adjust permissions on the video device if it remains locked:**
   ```bash
   sudo chmod 660 /dev/video0
   sudo chown root:video /dev/video0
   ```

### GStreamer Qualcomm Specifics
For the `qtiqmmfsrc` plugin to function seamlessly inside a container, you may need to map additional Qualcomm memory allocation devices depending on the BSP (Board Support Package) version.
If your pipeline crashes with memory buffer errors, uncomment the following in your `docker-compose.yml` under the `devices` block:
- `/dev/ion:/dev/ion`
- `/dev/dmabuf_heaps:/dev/dmabuf_heaps`

---

## 2. Deployment Profiles Orchestration

The `docker-compose.yml` utilizes architectural profiles to isolate environment behaviors without rewriting configuration code.

### Option A: Local Development (Windows / Standard Ubuntu)
This profile uses the local webcam, loads the PyTorch backend (`yolov8n-pose.pt`), and attempts to forward the OpenCV GUI back to your host machine.

```bash
docker compose --profile dev up --build
```
*Linux Note*: If running on an Ubuntu desktop and you want the OpenCV GUI window to spawn, you must temporarily expose your local X11 server to the container network:
```bash
xhost +local:docker
```

### Option B: Edge Production (Qualcomm Rubik Pi)
This profile forces `HEADLESS` mode, triggers the hardware-accelerated `GSTREAMER` pipeline, and loads the lightweight `ONNX` backend (`yolov8n-pose.onnx`) to conserve memory.

1. Ensure the optimized `yolov8n-pose.onnx` weight file is present in the project root directory alongside `Dockerfile`.
2. Build and launch the container as an isolated background daemon:
   ```bash
   docker compose --profile edge up -d --build
   ```

---

## 3. Persistent Telemetry Data Configuration
The `docker-compose.yml` utilizes a persistent named volume (`deskbot_metrics_volume`). By default, it maps to `/app/data` inside the container. 

**CRITICAL INTEGRATION STEP:**
To ensure historical metrics survive container rebuilds, ensure your `main.py` entrypoint is updated to point the `TelemetryLogger` to this specific mounted directory rather than the local working path:

```python
# In main.py
telemetry = TelemetryLogger('/app/data/deskbot_metrics.db')
```

---

## 4. Edge Container Management CLI Operations
- **Monitor the Headless Feed**: `docker logs -f deskbot_rubik`
- **Safely Stop the Daemon**: `docker compose --profile edge down`
- **Verify Camera Hardware Access Inside Sandbox**: `docker exec -it deskbot_rubik ls -l /dev/video0`
