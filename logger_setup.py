import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger():
    
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a logger
    # logger = logging.getLogger(script_name)
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.DEBUG)

    # Check if the logger already has handlers
    if not logger.hasHandlers():
        # Rotating file handler
        file_handler = RotatingFileHandler(log_dir / 'all.log', maxBytes=10*1024*1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(name)-15s - [%(filename)-18s:%(lineno)4d] - %(message)s"
        )
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")

        # Add formatters to handlers
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger