import multiprocessing
import time
from plyer import notification
import config

def voice_subprocess_worker(message_queue):
    """Runs as a completely independent OS process.
    
    Initializes and terminates the speech engine on-demand for every 
    individual message to prevent laptop driver lockups.
    """
    import pyttsx3  # Imported inside the process scope to ensure a clean sandbox environment

    while True:
        try:
            # Block until an alert message arrives through the pipeline
            text = message_queue.get()
            if text == "SIG_KILL":  # Clean shutdown handler
                break
                
            # Initialize, configure, speak, and completely tear down the engine instantly
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.say(text)
            engine.runAndWait()
            
            # Explicitly stop and close the audio driver instance to guarantee release
            engine.stop()
            del engine
            
        except Exception as e:
            print(f"[Audio Subprocess Error] Engine recycling exception: {e}")
            time.sleep(0.1)

class NotificationManager:
    def __init__(self):
        self.last_notified = {}
        
        # Instantiate an Inter-Process Communication (IPC) Queue
        self.msg_queue = multiprocessing.Queue()
        
        # Spin up the completely isolated voice engine process
        self.voice_process = multiprocessing.Process(
            target=voice_subprocess_worker, 
            args=(self.msg_queue,), 
            daemon=True
        )
        self.voice_process.start()

    def _speak(self, text):
        """Pushes text down the IPC pipeline to the independent voice worker."""
        self.msg_queue.put(text)

    def send_alert(self, alert_type, title, message):
        """Sends a notification if the cooldown period has passed."""
        current_time = time.time()
        last_time = self.last_notified.get(alert_type, 0)

        if current_time - last_time > config.NOTIFICATION_COOLDOWN:
            print(f"[{alert_type.upper()}] {title}: {message}")
            
            if config.ENABLE_DESKTOP_ALERTS:
                try:
                    notification.notify(
                        title=title,
                        message=message,
                        app_name="Desk Wellness Bot",
                        timeout=5
                    )
                except Exception as e:
                    print(f"Desktop notification failed: {e}")

            if config.ENABLE_VOICE_ALERTS:
                self._speak(message)

            self.last_notified[alert_type] = current_time

    def shutdown(self):
        """Safely terminates the voice subprocess context when exiting."""
        self.msg_queue.put("SIG_KILL")
        self.voice_process.join(timeout=1)