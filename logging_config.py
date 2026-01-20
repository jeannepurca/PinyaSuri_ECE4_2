#!/usr/bin/env python3
# logging_config.py

import logging
import config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.FLIGHT_LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("PinyaSuri")