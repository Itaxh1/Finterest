# finsight/__init__.py
"""
FinSight AI - Financial insights from news and market data.

This package provides tools for collecting, analyzing, and generating
insights from financial news and market data.
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

__version__ = "0.1.0"