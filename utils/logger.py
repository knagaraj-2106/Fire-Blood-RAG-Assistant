"""
Logger Utility
"""

import sys
from pathlib import Path
import logging
from config import LOG_DIR,LOG_FILE
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from config import LOG_DIR, LOG_FILE

Path(LOG_DIR).mkdir(exist_ok=True)

def setup_logger():
    logger = logging.getLogger('RAG')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"
    )
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()