#!/usr/bin/env python3
"""
Logging utilities for DSL model training.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "dsl_training",
    log_dir: Optional[str | Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (None = no file logging)
        level: Logging level
        console: Whether to log to console
    
    Returns:
        Configured logger
    """
    global _logger
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (None = return global logger)
    
    Returns:
        Logger instance
    """
    global _logger
    
    if name:
        return logging.getLogger(name)
    
    if _logger is None:
        _logger = setup_logger()
    
    return _logger


class TrainingLogger:
    """
    Training-specific logger with structured logging.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.step = 0
        self.epoch = 0
    
    def log_step(self, step: int, metrics: dict) -> None:
        """Log training step metrics."""
        self.step = step
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step:>6d} | {metrics_str}")
    
    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch metrics."""
        self.epoch = epoch
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch:>3d} | {metrics_str}")
    
    def log_eval(self, metrics: dict) -> None:
        """Log evaluation metrics."""
        self.logger.info("=" * 50)
        self.logger.info("Evaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.logger.info(f"  {k}: {v:.4f}")
            else:
                self.logger.info(f"  {k}: {v}")
        self.logger.info("=" * 50)
    
    def log_config(self, config: dict) -> None:
        """Log configuration."""
        self.logger.info("Configuration:")
        for section, values in config.items():
            if isinstance(values, dict):
                self.logger.info(f"  [{section}]")
                for k, v in values.items():
                    self.logger.info(f"    {k}: {v}")
            else:
                self.logger.info(f"  {section}: {values}")
