# finsight/analysis/sentiment_analyzer.py
"""
Module for analyzing sentiment in financial news.
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    logger.info("Downloading NLTK vader_lexicon...")
    nltk.download("vader_lexicon")


class SentimentAnalyzer:
    """Class for analyzing sentiment in text data."""

    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in text.
        """
        if not isinstance(text, str) or not text:
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            }

        # Get sentiment scores
        sentiment_scores = self.sia.polarity_scores(text)

        return {
            "compound": sentiment_scores["compound"],
            "positive": sentiment_scores["pos"],
            "negative": sentiment_scores["neg"],
            "neutral": sentiment_scores["neu"],
        }

    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of news articles.
        """
        result_df = news_df.copy()

        # Create sentiment columns
        result_df["sentiment_compound"] = 0.0
        result_df["sentiment_positive"] = 0.0
        result_df["sentiment_negative"] = 0.0
        result_df["sentiment_neutral"] = 0.0
        result_df["sentiment_label"] = ""

        for index, row in result_df.iterrows():
            title = row.get("title", "")
            content = row.get("content", "")

            # Combine title and content
            text = f"{title}. {content}"

            # Analyze sentiment
            sentiment = self.analyze_sentiment(text)

            # Update DataFrame
            result_df.at[index, "sentiment_compound"] = sentiment["compound"]
            result_df.at[index, "sentiment_positive"] = sentiment["positive"]
            result_df.at[index, "sentiment_negative"] = sentiment["negative"]
            result_df.at[index, "sentiment_neutral"] = sentiment["neutral"]

            # Add sentiment label
            if sentiment["compound"] >= 0.05:
                result_df.at[index, "sentiment_label"] = "positive"
            elif sentiment["compound"] <= -0.05:
                result_df.at[index, "sentiment_label"] = "negative"
            else:
                result_df.at[index, "sentiment_label"] = "neutral"

        # Add timestamp
        result_df["sentiment_analyzed_at"] = datetime.now()

        return result_df