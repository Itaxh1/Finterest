# finsight/data_collection/news_collector.py
"""
Module for collecting news data from various sources.
"""

import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv

from finsight.config import (
    NEWS_API_KEY,
    DEFAULT_NEWS_QUERIES,
    DEFAULT_DAYS_BACK
)

# Setup logging
logger = logging.getLogger(__name__)


class NewsCollector:
    """Class for collecting financial news from various sources."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NewsCollector.
        """
        self.api_key = api_key or NEWS_API_KEY

        if not self.api_key:
            logger.warning("No NewsAPI key provided or found in environment.")
            raise ValueError("NewsAPI key is required")

        self.newsapi = NewsApiClient(api_key=self.api_key)

    def collect_financial_news(
            self,
            query_terms: List[str] = None,
            days_back: int = None,
            language: str = "en",
            page_size: int = 100,
    ) -> pd.DataFrame:
        """
        Collect financial news articles based on query terms.
        """
        query_terms = query_terms or DEFAULT_NEWS_QUERIES
        days_back = days_back or DEFAULT_DAYS_BACK

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        all_articles = []

        for query in query_terms:
            try:
                response = self.newsapi.get_everything(
                    q=query,
                    from_param=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    language=language,
                    sort_by="publishedAt",
                    page_size=page_size,
                )

                if response["status"] == "ok":
                    all_articles.extend(response["articles"])
                    logger.info(
                        f"Collected {len(response['articles'])} articles for query: {query}"
                    )
                else:
                    logger.error(f"Error in API response for query: {query}")

            except Exception as e:
                logger.error(f"Error collecting news for query {query}: {str(e)}")

        news_df = pd.DataFrame(all_articles)

        # Add collection timestamp
        news_df["collected_at"] = datetime.now()

        return news_df