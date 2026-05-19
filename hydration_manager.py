import time
import config

class HydrationManager:
    def __init__(self, notifier):
        self.notifier = notifier
        self.last_drink_time = time.time()
        self.bottle_present_time = 0

    def update(self, is_drinking, bottle_detected, water_level):
        current_time = time.time()

        if is_drinking:
            self.last_drink_time = current_time

        time_since_drink = current_time - self.last_drink_time

        if time_since_drink > config.HYDRATION_REMINDER_INTERVAL:
            self.notifier.send_alert(
                "hydration", 
                "Hydration Reminder", 
                "It's been a while. Time to drink some water!"
            )
            self.last_drink_time = current_time - (config.HYDRATION_REMINDER_INTERVAL * 0.8)

        if not bottle_detected:
            self.bottle_present_time = 0
            if time_since_drink > (config.HYDRATION_REMINDER_INTERVAL * 0.5):
                 self.notifier.send_alert(
                    "no_bottle", 
                    "No Water Bottle", 
                    "Keep a water bottle nearby to stay hydrated."
                 )
        else:
            self.bottle_present_time += 1

        if bottle_detected and water_level is not None:
            if water_level < config.LOW_WATER_THRESHOLD_PERCENT:
                self.notifier.send_alert(
                    "low_water", 
                    "Refill Water", 
                    "Your water bottle is looking empty. Please refill it."
                )

        return time_since_drink         