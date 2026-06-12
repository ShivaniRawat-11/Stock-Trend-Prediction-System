import os
import logging
import sys
from typing import List

def setup_logging(log_dir: str = "logs", log_filename: str = "system.log") -> logging.Logger:
    """
    Sets up a standardized logging configuration that logs messages to both 
    the console and a log file. If writing to log file fails (e.g. read-only filesystem),
    falls back gracefully to console-only logging.

    Args:
        log_dir (str): Directory where log file will be saved. Defaults to 'logs'.
        log_filename (str): Name of the log file. Defaults to 'system.log'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("StockMarketTrendPrediction")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for log messages
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler - Wrap in try-except for read-only environment robustness
    file_logging_ok = False
    log_filepath = ""
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_filepath = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        file_logging_ok = True
    except OSError as e:
        logger.warning("File logging disabled: Unable to create or write logs directory/file (%s). Falling back to console-only logging.", e)

    if file_logging_ok:
        logger.info("Logging initialized. Logs are saved to %s", log_filepath)
    else:
        logger.info("Logging initialized in console-only mode.")
        
    return logger

def ensure_directories(dirs: List[str], logger: logging.Logger = None) -> None:
    """
    Ensures that a list of directories exists. Creates them if they do not.
    Catches exceptions if directory creation fails (e.g., read-only filesystem).

    Args:
        dirs (List[str]): List of directory paths to verify/create.
        logger (logging.Logger, optional): Logger instance to write progress. Defaults to None.
    """
    for directory in dirs:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                if logger:
                    logger.info("Created directory: %s", directory)
            else:
                if logger:
                    logger.debug("Directory already exists: %s", directory)
        except OSError as e:
            if logger:
                logger.warning("Could not create directory %s (might be read-only): %s", directory, e)
            else:
                sys.stderr.write(f"WARNING: Could not create directory {directory}: {e}\n")

