import logging
import sys

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
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger