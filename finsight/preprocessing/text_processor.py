# finsight/preprocessing/text_processor.py
"""
Module for preprocessing text data.
"""

import logging
import re
from typing import List, Optional, Union

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup logging
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")


class TextProcessor:
    """Class for preprocessing text data."""

    def __init__(self, language: str = "english"):
        """
        Initialize the text processor.
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data.
        """
        if not isinstance(text, str) or not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Remove special characters and numbers
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back to text
        preprocessed_text = " ".join(tokens)

        return preprocessed_text

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text.
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)

        # Tokenize
        tokens = word_tokenize(preprocessed_text)

        # Calculate frequency distribution
        freq_dist = nltk.FreqDist(tokens)

        # Get top keywords
        keywords = [word for word, _ in freq_dist.most_common(top_n)]

        return keywords