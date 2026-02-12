"""
Logging configuration module for WSI processing pipeline.

Provides centralized logging setup with consistent formatting.
"""

import logging


def configure_logging():
    """
    Configure global logging with timestamp, level, and message format.
    
    Sets up INFO-level logging suitable for production pipelines.
    Should be called at application startup before processing begins.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
