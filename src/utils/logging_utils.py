"""
Logging utilities for 360° Motion Analysis Pipeline.
"""
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"motion_analysis_{timestamp}.log")
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),      # Log to file
            logging.StreamHandler()             # Log to console
        ]
    )
    
    # Suppress OpenCV warnings
    logging.getLogger("cv2").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class ProgressLogger:
    """
    Utility class for logging processing progress.
    """
    
    def __init__(self, total_items, logger, description="Processing"):
        """
        Initialize progress logger.
        
        Args:
            total_items: Total number of items to process
            logger: Logger instance
            description: Description of the process
        """
        self.total_items = total_items
        self.logger = logger
        self.description = description
        self.start_time = datetime.now()
        self.last_update = 0
        
    def update(self, current_item, update_interval=100):
        """
        Update progress log.
        
        Args:
            current_item: Current item number
            update_interval: How often to log updates
        """
        if current_item % update_interval == 0 or current_item == self.total_items:
            elapsed = datetime.now() - self.start_time
            progress_pct = (current_item / self.total_items) * 100
            
            if current_item > 0:
                avg_time_per_item = elapsed.total_seconds() / current_item
                eta_seconds = avg_time_per_item * (self.total_items - current_item)
                eta = f"ETA: {eta_seconds/60:.1f}m"
            else:
                eta = "ETA: calculating..."
            
            self.logger.info(
                f"{self.description}: {progress_pct:.1f}% "
                f"({current_item}/{self.total_items}) - {eta}"
            )
            
    def complete(self):
        """Log completion message."""
        total_time = datetime.now() - self.start_time
        self.logger.info(
            f"{self.description} completed in {total_time.total_seconds():.1f} seconds"
        )