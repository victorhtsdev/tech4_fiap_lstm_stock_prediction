import logging
import os

class AppLogger:
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if os.getenv("ENABLE_LOGS", "false").lower() == "true" else logging.CRITICAL)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Create file handler
        log_file = os.getenv("LOG_FILE", "logs/app.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger