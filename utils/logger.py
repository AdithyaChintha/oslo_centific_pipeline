import logging
import sys
import os
from datetime import datetime

def get_logger(name: str = "ray_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger.

    Args:
        name (str): Logger name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # logger = logging.getLogger("AzurePipelineWrapper")
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)  # Hide HTTP request/response details
    logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)  # Show warnings and errors only
    logging.getLogger("azure.core").setLevel(logging.WARNING)  # General Azure core logging
    if not logger.handlers:
        # Console handler (for stdout redirection)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (for Ray workers and unified logging) - unique timestamped file
        os.makedirs('pipeline_logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'pipeline_logs/csam_pipeline_{timestamp}.log'
        file_handler = logging.FileHandler(log_filename, mode='w')  # Write mode for new file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Print log file location for reference
        print(f"📋 Pipeline logs will be saved to: {log_filename}")

    return logger