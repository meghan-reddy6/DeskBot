import pyttsx3
from plyer import notification
import threading
import time
import config

class NotificationManager:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) # Speech rate
        self.last_notified = {}

    def _speak(self, text):
        """Runs TTS in a separate thread to prevent blocking the video stream."""
        def run_tts():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run_tts, daemon=True).start()

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