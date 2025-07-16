# finsight/data_storage/db_connector.py
"""
Module for connecting to and interacting with databases.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime

import pymongo
from pymongo.database import Database
from pymongo.collection import Collection

from finsight.config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    NEWS_COLLECTION,
    MARKET_COLLECTION
)

# Setup logging
logger = logging.getLogger(__name__)


class MongoDBConnector:
    """Class for connecting to and interacting with MongoDB."""

    def __init__(
            self,
            uri: Optional[str] = None,
            database: Optional[str] = None,
    ):
        """
        Initialize the MongoDB connector.
        """
        self.uri = uri or MONGODB_URI
        self.database_name = database or MONGODB_DATABASE
        self.client = None
        self.db = None

    def connect(self) -> Database:
        """
        Connect to the MongoDB database.
        """
        try:
            self.client = pymongo.MongoClient(self.uri)
            self.db = self.client[self.database_name]
            logger.info(f"Connected to MongoDB database: {self.database_name}")
            return self.db
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def setup_collections(self) -> Dict[str, Collection]:
        """
        Set up the necessary collections with indexes.
        """
        if not self.db:
            self.connect()

        # News collection
        news_collection = self.db[NEWS_COLLECTION]
        news_collection.create_index([("publishedAt", pymongo.DESCENDING)])
        news_collection.create_index([("source.name", pymongo.ASCENDING)])
        news_collection.create_index([("title", pymongo.TEXT)])

        # Market data collection
        market_collection = self.db[MARKET_COLLECTION]
        market_collection.create_index([("ticker", pymongo.ASCENDING)])
        market_collection.create_index([("Date", pymongo.DESCENDING)])

        return {
            NEWS_COLLECTION: news_collection,
            MARKET_COLLECTION: market_collection,
        }

    def store_news_data(self, news_df: pd.DataFrame) -> int:
        """
        Store news data in MongoDB.
        """
        try:
            if not self.db:
                self.connect()

            news_collection = self.db[NEWS_COLLECTION]

            # Convert DataFrame to dictionary records
            news_records = news_df.to_dict('records')

            # Add timestamp
            for record in news_records:
                record['stored_at'] = datetime.now()

            # Insert into MongoDB
            result = news_collection.insert_many(news_records)
            inserted_count = len(result.inserted_ids)
            logger.info(f"Inserted {inserted_count} news articles into MongoDB")
            return inserted_count

        except Exception as e:
            logger.error(f"Error storing news data: {str(e)}")
            return 0

    def store_market_data(self, market_data: Dict[str, pd.DataFrame]) -> int:
        """
        Store market data in MongoDB.
        """
        try:
            if not self.db:
                self.connect()

            market_collection = self.db[MARKET_COLLECTION]
            total_inserted = 0

            for ticker, data in market_data.items():
                # Reset index to make date a column
                df = data.reset_index()

                # Convert DataFrame to dictionary records
                records = df.to_dict('records')

                # Add timestamp
                for record in records:
                    record['stored_at'] = datetime.now()

                # Insert into MongoDB
                result = market_collection.insert_many(records)
                inserted_count = len(result.inserted_ids)
                total_inserted += inserted_count
                logger.info(f"Inserted {inserted_count} records for {ticker} into MongoDB")

            return total_inserted

        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
            return 0

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")