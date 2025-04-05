# finsight/data_collection/__init__.py
"""
Data collection package for FinSight AI.
"""

from finsight.data_collection.news_collector import NewsCollector
from finsight.data_collection.market_collector import MarketDataCollector

__all__ = ['NewsCollector', 'MarketDataCollector']