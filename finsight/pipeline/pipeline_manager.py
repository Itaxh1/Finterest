# finsight/pipeline/pipeline_manager.py
"""
Module for managing data pipelines.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd

from finsight.data_collection import NewsCollector, MarketDataCollector
from finsight.data_storage import MongoDBConnector
from finsight.preprocessing import TextProcessor
from finsight.analysis import SentimentAnalyzer
from finsight.config import (
    DEFAULT_NEWS_QUERIES,
    DEFAULT_TICKERS,
    DEFAULT_DAYS_BACK
)

# Setup logging
logger = logging.getLogger(__name__)


class PipelineManager:
    """Class for managing data pipelines."""

    def __init__(self):
        """Initialize the pipeline manager."""
        self.news_collector = NewsCollector()
        self.market_collector = MarketDataCollector()
        self.db_connector = MongoDBConnector()
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()

    def run_data_collection_pipeline(
            self,
            news_queries: Optional[List[str]] = None,
            tickers: Optional[List[str]] = None,
            days_back: int = None,
            store_data: bool = True
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Run the data collection pipeline.
        """
        news_queries = news_queries or DEFAULT_NEWS_QUERIES
        tickers = tickers or DEFAULT_TICKERS
        days_back = days_back or DEFAULT_DAYS_BACK

        start_time = datetime.now()
        logger.info(f"Starting data collection pipeline at {start_time}")

        # Step 1: Collect news data
        logger.info("Collecting news data...")
        news_df = self.news_collector.collect_financial_news(
            query_terms=news_queries,
            days_back=days_back,
        )

        # Step 2: Collect market data
        logger.info("Collecting market data...")
        market_data = self.market_collector.collect_market_data(
            tickers=tickers,
            period=f"{days_back}d",
        )

        # Step 3: Store data if requested
        if store_data:
            logger.info("Storing collected data...")

            # Connect to database
            self.db_connector.connect()

            # Set up collections
            self.db_connector.setup_collections()

            # Store news data
            news_count = self.db_connector.store_news_data(news_df)
            logger.info(f"Stored {news_count} news articles")

            # Store market data
            market_count = self.db_connector.store_market_data(market_data)
            logger.info(f"Stored {market_count} market data records")

            # Close connection
            self.db_connector.close()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Data collection pipeline completed in {duration} seconds")

        return {
            "news_data": news_df,
            "market_data": market_data
        }

    def run_analysis_pipeline(
            self,
            news_df: Optional[pd.DataFrame] = None,
            store_results: bool = True
    ) -> pd.DataFrame:
        """
        Run the analysis pipeline on news data.
        """
        start_time = datetime.now()
        logger.info(f"Starting analysis pipeline at {start_time}")

        # If no news data provided, get it from the database
        if news_df is None:
            logger.info("No news data provided, fetching from database...")

            # Connect to database
            self.db_connector.connect()

            # TODO: Implement fetching news data from database
            # For now, return early with warning
            logger.warning("Fetching news data from database not implemented yet")
            return pd.DataFrame()

        # Step 1: Preprocess text
        logger.info("Preprocessing news text...")
        preprocessed_df = news_df.copy()

        # Apply text preprocessing
        preprocessed_df["preprocessed_title"] = preprocessed_df["title"].apply(
            self.text_processor.preprocess_text
        )
        preprocessed_df["preprocessed_content"] = preprocessed_df["content"].apply(
            self.text_processor.preprocess_text
        )

        # Step 2: Analyze sentiment
        logger.info("Analyzing sentiment...")
        result_df = self.sentiment_analyzer.analyze_news_dataframe(preprocessed_df)

        # Step 3: Store results if requested
        if store_results:
            logger.info("Storing analysis results...")

            # TODO: Implement storing analysis results in database
            logger.warning("Storing analysis results not implemented yet")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analysis pipeline completed in {duration} seconds")

        return result_df

    def run_full_pipeline(
            self,
            news_queries: Optional[List[str]] = None,
            tickers: Optional[List[str]] = None,
            days_back: int = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        """
        Run the full data pipeline.
        """
        # Step 1: Run data collection pipeline
        data = self.run_data_collection_pipeline(
            news_queries=news_queries,
            tickers=tickers,
            days_back=days_back,
            store_data=True
        )

        # Step 2: Run analysis pipeline
        analyzed_news = self.run_analysis_pipeline(
            news_df=data["news_data"],
            store_results=True
        )

        # Return results
        return {
            "news_data": analyzed_news,
            "market_data": data["market_data"]
        }