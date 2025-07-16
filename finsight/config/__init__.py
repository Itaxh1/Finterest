# finsight/config/__init__.py
"""
Configuration package for FinSight AI.
"""

from finsight.config.settings import *

__all__ = [
    'NEWS_API_KEY',
    'ALPHA_VANTAGE_API_KEY',
    'MONGODB_URI',
    'MONGODB_DATABASE',
    'NEWS_COLLECTION',
    'MARKET_COLLECTION',
    'DEFAULT_TICKERS',
    'DEFAULT_NEWS_QUERIES',
    'DEFAULT_DAYS_BACK',
    'DEFAULT_MARKET_PERIOD',
    'DEFAULT_MARKET_INTERVAL'
]