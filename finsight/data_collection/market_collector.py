# finsight/data_collection/market_collector.py
"""
Module for collecting market data for stocks and cryptocurrencies.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from finsight.config import (
    DEFAULT_TICKERS,
    DEFAULT_MARKET_PERIOD,
    DEFAULT_MARKET_INTERVAL
)

# Setup logging
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Class for collecting market data for stocks and cryptocurrencies."""

    def collect_market_data(
            self,
            tickers: List[str] = None,
            period: str = None,
            interval: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect market data for specified tickers.
        """
        tickers = tickers or DEFAULT_TICKERS
        period = period or DEFAULT_MARKET_PERIOD
        interval = interval or DEFAULT_MARKET_INTERVAL

        market_data = {}

        for ticker in tickers:
            try:
                # Download data
                data = yf.download(ticker, period=period, interval=interval)

                # Add ticker to the data
                data["ticker"] = ticker

                # Add to dictionary
                market_data[ticker] = data

                logger.info(f"Collected {len(data)} records for {ticker}")

            except Exception as e:
                logger.error(f"Error collecting data for {ticker}: {str(e)}")

        return market_data