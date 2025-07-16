# finsight/config/settings.py
"""
Settings module for the FinSight AI project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Database settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "finsight")

# Collection settings
NEWS_COLLECTION = "news_articles"
MARKET_COLLECTION = "market_data"

# Default tickers to track
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META",
    "BTC-USD", "ETH-USD", "SOL-USD"
]

# Default news query terms
DEFAULT_NEWS_QUERIES = [
    "stock market", "finance", "cryptocurrency", "bitcoin",
    "Tesla stock", "Apple stock", "Amazon stock"
]

# Data collection settings
DEFAULT_DAYS_BACK = 30
DEFAULT_MARKET_PERIOD = "3mo"
DEFAULT_MARKET_INTERVAL = "1d"