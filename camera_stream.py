import cv2
import threading
import queue
import time
import os
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CrossPlatformCapture:
    """
    A cross-platform, threaded video capture module that dynamically switches between
    hardware-accelerated GStreamer pipelines (Qualcomm/Rubik Pi) and native OS backends.
    """
    def __init__(self, camera_index=0, width=1280, height=720, framerate=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.framerate = framerate
        
        self.cap = None
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = None
        
        self._initialize_camera()

    def _initialize_camera(self):
        """Initializes the VideoCapture object based on the environment."""
        camera_type = os.getenv("CAMERA_TYPE", "").upper()
        
        if camera_type == "GSTREAMER":
            logger.info("Initializing Qualcomm/Rubik Pi GStreamer pipeline...")
            # 'videoconvert' automatically handles the NV12 to BGR color space conversion
            # 'appsink drop=true max-buffers=1' ensures the sink drops stale frames if the pipeline backs up
            gstreamer_pipeline = (
                f"qtiqmmfsrc camera={self.camera_index} ! "
                f"video/x-raw,format=NV12,width={self.width},height={self.height},framerate={self.framerate}/1 ! "
                f"queue ! videoconvert ! appsink drop=true max-buffers=1"
            )
            self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            
        else:
            system = platform.system().lower()
            logger.info(f"Initializing native backend for platform: {system}")
            
            if system == "windows":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            elif system == "linux":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
                
            # Request specific hardware properties for native backends
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {self.camera_index}. Please check device connections and permissions.")
            
        logger.info("Camera initialized successfully.")

    def start(self):
        """Starts the background thread for frame ingestion."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Capture thread is already running.")
            return self
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info("Camera stream background thread started.")
        return self

    def _update(self):
        """Internal thread target that pumps frames from the camera into the queue."""
        consecutive_errors = 0
        max_errors = 10
        
        while not self._stop_event.is_set():
            if not self.cap.isOpened():
                logger.error("Camera connection lost.")
                break
                
            ret, frame = self.cap.read()
            if not ret:
                consecutive_errors += 1
                logger.warning(f"Failed to grab frame. Error count: {consecutive_errors}/{max_errors}")
                if consecutive_errors >= max_errors:
                    logger.error("Max consecutive frame drop errors reached. Stopping capture.")
                    break
                time.sleep(0.01)
                continue
                
            consecutive_errors = 0
            
            # Non-blocking put: if queue is full, drop the old frame and replace it with the freshest one
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait() # pop the stale frame
                    self._frame_queue.put_nowait(frame) # push the fresh frame
                except queue.Empty:
                    pass

    def read(self):
        """
        Reads the most recent frame from the queue.
        Returns a tuple (status, frame), matching the standard cv2.VideoCapture API.
        """
        try:
            # Block for a short time to wait for a frame if the queue is empty
            frame = self._frame_queue.get(timeout=2.0)
            return True, frame
        except queue.Empty:
            logger.error("Timeout waiting for frame from camera stream.")
            return False, None

    def stop(self):
        """Stops the background thread and releases camera resources."""
        logger.info("Stopping camera stream...")
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            
        if self.cap is not None:
            self.cap.release()
            
        logger.info("Camera stream stopped and resources released.")

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point to ensure cleanup."""
        self.stop()

# ==========================================
# Usage Example
# ==========================================
if __name__ == "__main__":
    # Simulate setting the environment variable for testing purposes
    # os.environ["CAMERA_TYPE"] = "GSTREAMER"  # Uncomment to test GStreamer path
    
    print("Starting cross-platform capture test...")
    try:
        # Context manager handles startup and graceful thread shutdown
        with CrossPlatformCapture(camera_index=0) as cam:
            frames_read = 0
            start_time = time.time()
            
            # Read frames for 5 seconds
            while time.time() - start_time < 5.0:
                ret, frame = cam.read()
                if ret:
                    # Your processing/inference goes here!
                    frames_read += 1
                    cv2.imshow("CrossPlatformCapture Test", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to read frame.")
                    break
                    
            duration = time.time() - start_time
            fps = frames_read / duration
            print(f"Test completed. Read {frames_read} frames in {duration:.2f}s (~{fps:.2f} FPS).")
            
    except Exception as e:
        print(f"Capture Error: {e}")
    finally:
        cv2.destroyAllWindows()
