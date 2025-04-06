import os
import time
import datetime
import json
import argparse
import re
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import tweepy
import praw
from googleapiclient.discovery import build
from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING, TEXT
from bson.objectid import ObjectId
import yfinance as yf
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from collections import Counter
from tqdm import tqdm
import requests

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load environment variables
load_dotenv()

# Configuration - Store these in .env file in production
SPOTIFY_CONFIG = {
    "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
    "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET")
}

TWITTER_CONFIG = {
    "bearer_token": os.getenv("TWITTER_BEARER_TOKEN")
}

REDDIT_CONFIG = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "username": os.getenv("REDDIT_USERNAME"),
    "password": os.getenv("REDDIT_PASSWORD")
}

YOUTUBE_CONFIG = {
    "api_key": os.getenv("YOUTUBE_API_KEY")
}

# MongoDB Configuration
MONGODB_CONFIG = {
    "connection_string": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
    "database_name": os.getenv("MONGODB_DB", "social_media_finance"),
}

# Common ticker mappings for popular companies
TICKER_MAPPINGS = {
    "TESLA": "TSLA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "NVIDIA": "NVDA"
}

class TextAnalyzer:
    """Class for text analysis including sentiment and context extraction"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add finance-specific stopwords
        self.stop_words.update(['company', 'stock', 'market', 'price', 'share', 'shares'])
        
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        if not text:
            return {"polarity": 0, "subjectivity": 0, "sentiment": "neutral"}
            
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    
    def preprocess_text(self, text):
        """Preprocess text for topic modeling"""
        if not text:
            return []
            
        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords, punctuation, and short words
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words 
                 and token.isalpha() 
                 and len(token) > 2]
                 
        return tokens
    
    def extract_keywords(self, text, num_keywords=5):
        """Extract keywords from text using frequency analysis"""
        if not text or len(text) < 50:
            return []
            
        # Preprocess text
        tokens = self.preprocess_text(text)
        if not tokens:
            return []
            
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Get most common words
        keywords = [word for word, count in word_freq.most_common(num_keywords)]
        
        return keywords
    
    def extract_context(self, text, num_topics=3):
        """Extract context/topics from text"""
        if not text or len(text) < 50:  # Skip very short texts
            return {"topics": [], "keywords": []}
            
        # Preprocess text
        tokens = self.preprocess_text(text)
        if not tokens:
            return {"topics": [], "keywords": []}
            
        # Create dictionary and corpus
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=min(num_topics, 3),  # Limit to 3 topics max
            passes=10,
            alpha='auto'
        )
        
        # Extract topics
        topics = []
        for topic_id, topic in lda_model.print_topics():
            # Extract words from the topic string
            words = re.findall(r'"([^"]*)"', topic)
            topics.append(words)
        
        # Extract keywords using our custom method instead of gensim.summarization
        keywords = self.extract_keywords(text)
            
        return {
            "topics": topics,
            "keywords": keywords
        }
    
    def deep_analyze_text(self, text, title=""):
        """Perform a deeper analysis of text content"""
        if not text or len(text) < 100:
            return {
                "summary": "",
                "main_entities": [],
                "sentiment": self.analyze_sentiment(text),
                "context": self.extract_context(text)
            }
        
        # Combine title and text for better context
        full_text = f"{title} {text}" if title else text
        
        # Extract entities (people, organizations, etc.)
        entities = []
        words = word_tokenize(full_text)
        pos_tags = nltk.pos_tag(words)
        
        # Look for proper nouns as potential entities
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NNP') and word.lower() not in self.stop_words and len(word) > 1:
                entities.append(word)
        
        # Get unique entities
        unique_entities = list(set(entities))
        
        # Generate a simple summary (first 2-3 sentences)
        sentences = nltk.sent_tokenize(text)
        summary = " ".join(sentences[:min(3, len(sentences))])
        
        return {
            "summary": summary,
            "main_entities": unique_entities[:10],  # Top 10 entities
            "sentiment": self.analyze_sentiment(full_text),
            "context": self.extract_context(full_text)
        }

class CompanyInfoRetriever:
    """Class for retrieving basic company information"""
    
    def get_company_info(self, company_name):
        """Get basic company information"""
        try:
            # Check if we have a known mapping for this ticker
            ticker_to_use = company_name
            if company_name.upper() in TICKER_MAPPINGS:
                ticker_to_use = TICKER_MAPPINGS[company_name.upper()]
                print(f"📊 Using ticker symbol: {ticker_to_use}")
            
            # Get basic company info without stock prices
            stock = yf.Ticker(ticker_to_use)
            info = stock.info
            
            company_data = {
                "name": info.get("shortName", company_name),
                "ticker": ticker_to_use,
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "country": info.get("country", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "ceo": info.get("companyOfficers", [{}])[0].get("name", "") if info.get("companyOfficers") else ""
            }
            
            return company_data
        except Exception as e:
            print(f"❌ Error retrieving company info for {company_name}: {str(e)}")
            return {
                "name": company_name,
                "ticker": company_name,
                "description": ""
            }
    
    def get_news(self, company_name, limit=5):
        """Get news articles for a company using Yahoo Finance"""
        try:
            # Check if we have a known mapping for this ticker
            ticker_to_use = company_name
            if company_name.upper() in TICKER_MAPPINGS:
                ticker_to_use = TICKER_MAPPINGS[company_name.upper()]
                print(f"📰 Using ticker symbol for news: {ticker_to_use}")
            
            ticker = yf.Ticker(ticker_to_use)
            news = ticker.news
            
            formatted_news = []
            for article in news[:limit]:
                formatted_news.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "publish_time": datetime.datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                    "type": "news_article",
                    "platform": "yahoo_finance"
                })
                
            return formatted_news
        except Exception as e:
            print(f"❌ Error retrieving news for {company_name}: {str(e)}")
            return []

class FinancialDataRetriever:
    """Class for retrieving financial data"""
    
    def get_stock_data(self, ticker, period="1mo"):
        """Get stock data for a ticker"""
        try:
            # Check if we have a known mapping for this ticker
            if ticker.upper() in TICKER_MAPPINGS:
                ticker = TICKER_MAPPINGS[ticker.upper()]
                print(f"📈 Using ticker symbol: {ticker}")
            
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            # Get company info
            info = stock.info
            
            # Format the data
            stock_data = {
                "ticker": ticker,
                "company_name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "price_history": [],
                "price_change_percent": 0,
                "volume": info.get("volume", 0),
                "average_volume": info.get("averageVolume", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "target_price": info.get("targetMeanPrice", 0),
                "recommendation": info.get("recommendationKey", "")
            }
            
            # Format price history
            if not hist.empty:
                for date, row in hist.iterrows():
                    stock_data["price_history"].append({
                        "date": date.strftime("%Y-%m-%d"),
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"]
                    })
                
                # Calculate price change percentage
                if len(stock_data["price_history"]) >= 2:
                    first_price = stock_data["price_history"][0]["close"]
                    last_price = stock_data["price_history"][-1]["close"]
                    stock_data["price_change_percent"] = ((last_price - first_price) / first_price) * 100
            
            return stock_data
        except Exception as e:
            print(f"❌ Error retrieving stock data for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "error": str(e)
            }
    
    def get_news(self, company_name, limit=5):
        """Get news articles for a company using Yahoo Finance"""
        try:
            # Check if we have a known mapping for this ticker
            ticker_to_use = company_name
            if company_name.upper() in TICKER_MAPPINGS:
                ticker_to_use = TICKER_MAPPINGS[company_name.upper()]
                print(f"📰 Using ticker symbol for news: {ticker_to_use}")
            
            ticker = yf.Ticker(ticker_to_use)
            news = ticker.news
            
            formatted_news = []
            for article in news[:limit]:
                formatted_news.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "publish_time": datetime.datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                    "type": "news_article",
                    "platform": "yahoo_finance"
                })
                
            return formatted_news
        except Exception as e:
            print(f"❌ Error retrieving news for {company_name}: {str(e)}")
            return []

class SocialMediaAggregator:
    def __init__(self, company_name, person_name=None, use_mongodb=True):
        self.company_name = company_name
        self.person_name = person_name
        self.use_mongodb = use_mongodb
        
        # Construct query based on inputs
        if person_name:
            self.query = f"{company_name} OR {person_name}"
            self.query_description = f"Content related to {company_name} or {person_name}"
        else:
            self.query = company_name
            self.query_description = f"Content related to {company_name}"
        
        self.data = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "source": "finance_aggregation",
            "query": self.query,
            "query_description": self.query_description,
            "company_name": company_name,
            "person_name": person_name,
            "company_info": {},
            "platforms": {
                "spotify": [],
                "twitter": [],
                "reddit": [],
                "youtube": [],
                "news": []
            },
            "all_responses": []  # Unified array for all responses
        }
        
        # Initialize text analyzer
        self.text_analyzer = TextAnalyzer()
        
        # Initialize company info retriever
        self.company_retriever = CompanyInfoRetriever()
        
        # Initialize MongoDB connection if requested
        self.mongo_client = None
        self.db = None
        if self.use_mongodb:
            self.initialize_mongodb()

    def initialize_mongodb(self):
        """Initialize MongoDB connection and create indexes"""
        try:
            print("🔄 Connecting to MongoDB...")
            # Set a shorter timeout for MongoDB connection
            self.mongo_client = MongoClient(
                MONGODB_CONFIG["connection_string"],
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            
            # Test the connection
            self.mongo_client.server_info()
            
            self.db = self.mongo_client[MONGODB_CONFIG["database_name"]]
            
            # Create collections if they don't exist
            if "aggregations" not in self.db.list_collection_names():
                self.db.create_collection("aggregations")
            
            if "content_items" not in self.db.list_collection_names():
                content_items = self.db.create_collection("content_items")
                
                # Create indexes for efficient querying
                content_items.create_indexes([
                    IndexModel([("platform", ASCENDING)], name="platform_idx"),
                    IndexModel([("type", ASCENDING)], name="type_idx"),
                    IndexModel([("timestamp", DESCENDING)], name="timestamp_idx"),
                    IndexModel([("aggregation_id", ASCENDING)], name="aggregation_id_idx"),
                    IndexModel([("company_name", ASCENDING)], name="company_name_idx"),
                    IndexModel([("person_name", ASCENDING)], name="person_name_idx"),
                    IndexModel([("sentiment.sentiment", ASCENDING)], name="sentiment_idx"),
                    IndexModel([("title", TEXT), ("description", TEXT), ("text", TEXT), ("body", TEXT)], 
                              name="content_text_idx")
                ])
            
            if "company_info" not in self.db.list_collection_names():
                company_info = self.db.create_collection("company_info")
                
                # Create indexes for company info
                company_info.create_indexes([
                    IndexModel([("ticker", ASCENDING)], name="ticker_idx", unique=True),
                    IndexModel([("name", ASCENDING)], name="company_name_idx")
                ])
            
            print("✅ MongoDB connection established and collections initialized")
        except Exception as e:
            print(f"❌ Error initializing MongoDB: {str(e)}")
            print("⚠️ Continuing without MongoDB support")
            self.use_mongodb = False
            self.mongo_client = None
            self.db = None

    # Helper to add to unified array
    def add_to_all_responses(self, platform, item):
        unified_item = {
            "platform": platform,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "content": item,
            "original_data": item,
            "company_name": self.company_name,
            "person_name": self.person_name
        }
        self.data["all_responses"].append(unified_item)
        return unified_item

    # Fetch company information
    def fetch_company_info(self):
        print(f"\n🏢 Fetching company information for {self.company_name}...")
        try:
            # Get basic company info
            company_info = self.company_retriever.get_company_info(self.company_name)
            
            # Store company info
            self.data["company_info"] = company_info
            
            # Get news articles
            news_articles = self.company_retriever.get_news(self.company_name)
            
            # Add sentiment and context analysis to news
            for article in news_articles:
                article["sentiment"] = self.text_analyzer.analyze_sentiment(article["title"])
                article["context"] = self.text_analyzer.extract_context(article["title"])
                self.data["platforms"]["news"].append(article)
                self.add_to_all_responses("news", article)
            
            print(f"✅ Company information collected for {self.company_name}")
            print(f"   Company: {company_info.get('name', 'N/A')}")
            print(f"   Sector: {company_info.get('sector', 'N/A')}")
            print(f"   Industry: {company_info.get('industry', 'N/A')}")
            
            print(f"✅ News articles collected: {len(news_articles)} items")
            
        except Exception as error:
            print(f"❌ Error fetching company information: {str(error)}")

    # Spotify data collection with transcript analysis
    def fetch_spotify_data(self):
        print("\n🎧 Fetching Spotify Podcasts and Transcripts...")
        try:
            spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CONFIG["client_id"],
                client_secret=SPOTIFY_CONFIG["client_secret"]
            ))
            
            # Optimized query for finance podcasts related to the company
            podcast_results = spotify.search(
                q=f"finance investing {self.company_name}", 
                type="show", 
                limit=5,
                market="US"
            )
            
            # Create progress bar for podcast episodes
            total_shows = len(podcast_results["shows"]["items"])
            with tqdm(total=total_shows, desc="Podcasts", unit="show") as pbar:
                for show in podcast_results["shows"]["items"]:
                    show_id = show["id"]
                    show_name = show["name"]
                    episodes = spotify.show_episodes(show_id, limit=10)
                    
                    for ep in episodes["items"]:
                        # Check if episode description or title contains query terms
                        query_terms = [term.strip().lower() for term in self.query.split("OR")]
                        matches_query = any(
                            term in ep["name"].lower() or term in ep["description"].lower() 
                            for term in query_terms
                        )
                        
                        if matches_query:
                            # Perform deep analysis on the podcast description
                            # This simulates transcript analysis since actual transcripts aren't available via API
                            deep_analysis = self.text_analyzer.deep_analyze_text(
                                ep["description"], 
                                title=ep["name"]
                            )
                            
                            # Try to find additional content about this episode online
                            additional_content = self.find_podcast_transcript(
                                show_name, 
                                ep["name"], 
                                self.company_name
                            )
                            
                            spotify_item = {
                                "podcast": show_name,
                                "title": ep["name"],
                                "description": ep["description"],
                                "release_date": ep["release_date"],
                                "audio": ep["audio_preview_url"],
                                "url": ep["external_urls"]["spotify"],
                                "type": "podcast",
                                "platform": "spotify",
                                "company_name": self.company_name,
                                "person_name": self.person_name,
                                "sentiment": deep_analysis["sentiment"],
                                "context": deep_analysis["context"],
                                "summary": deep_analysis["summary"],
                                "main_entities": deep_analysis["main_entities"],
                                "additional_content": additional_content
                            }
                            
                            self.data["platforms"]["spotify"].append(spotify_item)
                            self.add_to_all_responses("spotify", spotify_item)
                    
                    pbar.update(1)
            
            print(f"✅ Spotify data collected: {len(self.data['platforms']['spotify'])} items")
        except Exception as error:
            print(f"❌ Error fetching Spotify data: {str(error)}")
    
    def find_podcast_transcript(self, podcast_name, episode_name, company_name):
        """Attempt to find podcast transcript or additional content"""
        try:
            # This is a simulated function since actual transcript retrieval would require
            # specialized services or web scraping which is beyond the scope
            
            # For now, we'll return a placeholder with search terms that would be used
            return {
                "transcript_found": False,
                "search_terms": f"{podcast_name} {episode_name} {company_name} transcript",
                "potential_sources": ["podcast website", "YouTube captions", "transcript services"]
            }
        except Exception:
            return {"transcript_found": False}

    # Twitter data collection with focus on top tweets and conversations
    def fetch_twitter_data(self):
        print("\n🐦 Fetching Top Twitter Tweets and Conversations...")
        try:
            client = tweepy.Client(bearer_token=TWITTER_CONFIG["bearer_token"])
            
            # Optimized Twitter query with additional filters
            twitter_query = f"{self.query} lang:en -is:retweet"
            
            try:
                # Get recent tweets with high engagement
                twitter_response = client.search_recent_tweets(
                    query=twitter_query,
                    tweet_fields=["created_at", "public_metrics", "author_id", "entities", "conversation_id"],
                    expansions=["author_id", "referenced_tweets.id"],
                    max_results=100
                )
                
                if twitter_response.data:
                    # Sort tweets by engagement (likes + retweets)
                    sorted_tweets = sorted(
                        twitter_response.data,
                        key=lambda x: (
                            x.public_metrics.get("like_count", 0) + 
                            x.public_metrics.get("retweet_count", 0)
                        ) if hasattr(x, "public_metrics") else 0,
                        reverse=True
                    )
                    
                    # Take only top 20 tweets
                    top_tweets = sorted_tweets[:20]
                    
                    # Create progress bar for tweets
                    with tqdm(total=len(top_tweets), desc="Top Tweets", unit="tweet") as pbar:
                        for tweet in top_tweets:
                            # Extract links if available
                            links = []
                            if hasattr(tweet, "entities") and tweet.entities and "urls" in tweet.entities:
                                links = [url["expanded_url"] for url in tweet.entities["urls"]]
                            
                            # Get conversation replies for this tweet
                            conversation_tweets = []
                            try:
                                if hasattr(tweet, "conversation_id"):
                                    # Get replies in the conversation
                                    replies = client.search_recent_tweets(
                                        query=f"conversation_id:{tweet.conversation_id}",
                                        tweet_fields=["created_at", "public_metrics", "author_id"],
                                        max_results=10
                                    )
                                    
                                    if replies.data:
                                        for reply in replies.data:
                                            if reply.id != tweet.id:  # Skip the original tweet
                                                conversation_tweets.append({
                                                    "id": str(reply.id),
                                                    "text": reply.text,
                                                    "created_at": reply.created_at.isoformat() if hasattr(reply.created_at, "isoformat") else str(reply.created_at),
                                                    "metrics": reply.public_metrics if hasattr(reply, "public_metrics") else {}
                                                })
                            except Exception as e:
                                print(f"  ⚠️ Could not fetch conversation: {str(e)}")
                            
                            # Perform deep analysis on the tweet and its conversation
                            all_text = tweet.text
                            for reply in conversation_tweets:
                                all_text += " " + reply["text"]
                            
                            deep_analysis = self.text_analyzer.deep_analyze_text(all_text)
                            
                            twitter_item = {
                                "id": str(tweet.id),
                                "text": tweet.text,
                                "created_at": tweet.created_at.isoformat() if hasattr(tweet.created_at, "isoformat") else str(tweet.created_at),
                                "author_id": str(tweet.author_id),
                                "metrics": tweet.public_metrics if hasattr(tweet, "public_metrics") else {},
                                "links": links,
                                "type": "tweet",
                                "platform": "twitter",
                                "company_name": self.company_name,
                                "person_name": self.person_name,
                                "sentiment": deep_analysis["sentiment"],
                                "context": deep_analysis["context"],
                                "conversation_tweets": conversation_tweets,
                                "conversation_summary": deep_analysis["summary"],
                                "main_entities": deep_analysis["main_entities"]
                            }
                            
                            self.data["platforms"]["twitter"].append(twitter_item)
                            self.add_to_all_responses("twitter", twitter_item)
                            pbar.update(1)
                    
                    print(f"✅ Twitter data collected: {len(self.data['platforms']['twitter'])} top tweets with {sum(len(t.get('conversation_tweets', [])) for t in self.data['platforms']['twitter'])} replies")
                else:
                    print("⚠️ No Twitter data found.")
            except tweepy.TweepyException as e:
                if "429" in str(e):
                    print("⚠️ Twitter API rate limit reached. Skipping Twitter data collection.")
                else:
                    raise e
                
        except Exception as error:
            print(f"❌ Error fetching Twitter data: {str(error)}")

    # Reddit data collection
    def fetch_reddit_data(self):
        print("\n👽 Fetching Reddit Posts...")
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CONFIG["client_id"],
                client_secret=REDDIT_CONFIG["client_secret"],
                user_agent="finance-aggregator-app",
                username=REDDIT_CONFIG["username"],
                password=REDDIT_CONFIG["password"]
            )
            
            reddit_keywords = [k.strip() for k in self.query.split("OR")]
            # Expanded list of finance subreddits
            subreddits = [
                "stocks", "investing", "wallstreetbets", "finance", 
                "SecurityAnalysis", "Economics", "StockMarket", "options"
            ]
            
            # Create progress bar for subreddits
            with tqdm(total=len(subreddits), desc="Subreddits", unit="sub") as pbar:
                for sub in subreddits:
                    subreddit = reddit.subreddit(sub)
                    
                    # Get both hot and new posts for better coverage
                    hot_posts = list(subreddit.hot(limit=100))
                    new_posts = list(subreddit.new(limit=200))
                    posts = hot_posts + new_posts
                    
                    matching_posts = 0
                    for submission in posts:
                        if any(
                            k.lower() in submission.title.lower() or 
                            k.lower() in submission.selftext.lower() 
                            for k in reddit_keywords
                        ):
                            matching_posts += 1
                            
                            # Get top comments for this post
                            submission.comment_sort = 'top'
                            submission.comments.replace_more(limit=0)  # Don't fetch deeper comment trees
                            top_comments = submission.comments[:10]  # Get top 10 comments
                            
                            comments_data = []
                            for comment in top_comments:
                                comments_data.append({
                                    "id": comment.id,
                                    "author": comment.author.name if comment.author else "[deleted]",
                                    "body": comment.body,
                                    "score": comment.score,
                                    "created_utc": datetime.datetime.fromtimestamp(comment.created_utc).isoformat()
                                })
                            
                            # Combine post and comments for deeper analysis
                            full_text = f"{submission.title} {submission.selftext}"
                            for comment in comments_data:
                                full_text += " " + comment["body"]
                            
                            deep_analysis = self.text_analyzer.deep_analyze_text(full_text, title=submission.title)
                            
                            reddit_item = {
                                "subreddit": sub,
                                "title": submission.title,
                                "body": submission.selftext,
                                "url": f"https://reddit.com{submission.permalink}",
                                "score": submission.score,
                                "num_comments": submission.num_comments,
                                "created_utc": datetime.datetime.fromtimestamp(submission.created_utc).isoformat(),
                                "author": submission.author.name if submission.author else "[deleted]",
                                "type": "reddit_post",
                                "platform": "reddit",
                                "company_name": self.company_name,
                                "person_name": self.person_name,
                                "sentiment": deep_analysis["sentiment"],
                                "context": deep_analysis["context"],
                                "top_comments": comments_data,
                                "summary": deep_analysis["summary"],
                                "main_entities": deep_analysis["main_entities"]
                            }
                            
                            self.data["platforms"]["reddit"].append(reddit_item)
                            self.add_to_all_responses("reddit", reddit_item)
                    
                    pbar.set_postfix({"matches": matching_posts})
                    pbar.update(1)
            
            print(f"✅ Reddit data collected: {len(self.data['platforms']['reddit'])} posts with {sum(len(p.get('top_comments', [])) for p in self.data['platforms']['reddit'])} comments")
        except Exception as error:
            print(f"❌ Error fetching Reddit data: {str(error)}")

    # YouTube data collection
    def fetch_youtube_data(self):
        print("\n📺 Fetching YouTube Videos...")
        try:
            youtube = build("youtube", "v3", developerKey=YOUTUBE_CONFIG["api_key"])
            
            # Calculate date 30 days ago for filtering
            thirty_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat() + "Z"
            
            # Optimized YouTube query with additional parameters
            request = youtube.search().list(
                q=self.query,
                part="snippet",
                type="video",
                maxResults=50,
                order="relevance",
                publishedAfter=thirty_days_ago,
                relevanceLanguage="en",
                videoDuration="medium",  # Medium length videos
                videoDefinition="high"  # High definition
            )
            
            response = request.execute()
            
            # Create progress bar for YouTube videos
            with tqdm(total=len(response["items"]), desc="YouTube videos", unit="video") as pbar:
                for item in response["items"]:
                    # Get video details for more metrics
                    video_id = item["id"]["videoId"]
                    video_details = youtube.videos().list(
                        part="statistics,contentDetails",
                        id=video_id
                    ).execute()
                    
                    statistics = {}
                    content_details = {}
                    
                    if video_details["items"]:
                        if "statistics" in video_details["items"][0]:
                            statistics = video_details["items"][0]["statistics"]
                        if "contentDetails" in video_details["items"][0]:
                            content_details = video_details["items"][0]["contentDetails"]
                    
                    # Try to get video comments
                    comments = []
                    try:
                        comments_response = youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=10,
                            order="relevance"
                        ).execute()
                        
                        if "items" in comments_response:
                            for comment_item in comments_response["items"]:
                                comment_snippet = comment_item["snippet"]["topLevelComment"]["snippet"]
                                comments.append({
                                    "author": comment_snippet["authorDisplayName"],
                                    "text": comment_snippet["textDisplay"],
                                    "likes": comment_snippet["likeCount"],
                                    "published_at": comment_snippet["publishedAt"]
                                })
                    except Exception as e:
                        # Comments might be disabled
                        pass
                    
                    # Analyze sentiment and extract context from title, description and comments
                    combined_text = f"{item['snippet']['title']} {item['snippet']['description']}"
                    for comment in comments:
                        combined_text += " " + comment["text"]
                    
                    deep_analysis = self.text_analyzer.deep_analyze_text(
                        combined_text, 
                        title=item['snippet']['title']
                    )
                    
                    youtube_item = {
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "publish_time": item["snippet"]["publishedAt"],
                        "description": item["snippet"]["description"],
                        "video_id": video_id,
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                        "views": int(statistics.get("viewCount", 0)),
                        "likes": int(statistics.get("likeCount", 0)),
                        "comments": int(statistics.get("commentCount", 0)),
                        "duration": content_details.get("duration", ""),
                        "type": "youtube_video",
                        "platform": "youtube",
                        "company_name": self.company_name,
                        "person_name": self.person_name,
                        "sentiment": deep_analysis["sentiment"],
                        "context": deep_analysis["context"],
                        "top_comments": comments,
                        "summary": deep_analysis["summary"],
                        "main_entities": deep_analysis["main_entities"]
                    }
                    
                    self.data["platforms"]["youtube"].append(youtube_item)
                    self.add_to_all_responses("youtube", youtube_item)
                    pbar.update(1)
            
            print(f"✅ YouTube data collected: {len(self.data['platforms']['youtube'])} videos with {sum(len(v.get('top_comments', [])) for v in self.data['platforms']['youtube'])} comments")
        except Exception as error:
            print(f"❌ Error fetching YouTube data: {str(error)}")

    # Export data to JSON file
    def export_to_json(self, filename=None):
        """Export the collected data to a JSON file"""
        if not filename:
            # Create filename based on company and timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            company_slug = self.company_name.lower().replace(" ", "_")
            filename = f"{company_slug}_{timestamp}.json"
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"✅ Data exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error exporting data to JSON: {str(e)}")
            return False

    # Store data in MongoDB
    def store_in_mongodb(self):
        """Store the collected data in MongoDB with optimized structure"""
        if not self.use_mongodb or not self.db:
            print("❌ MongoDB connection not available")
            return False
        
        try:
            # Create aggregation record
            aggregation = {
                "timestamp": datetime.datetime.utcnow(),
                "query": self.query,
                "query_description": self.query_description,
                "company_name": self.company_name,
                "person_name": self.person_name,
                "source": "finance_aggregation",
                "platform_counts": {
                    "spotify": len(self.data["platforms"]["spotify"]),
                    "twitter": len(self.data["platforms"]["twitter"]),
                    "reddit": len(self.data["platforms"]["reddit"]),
                    "youtube": len(self.data["platforms"]["youtube"]),
                    "news": len(self.data["platforms"]["news"])
                },
                "total_items": len(self.data["all_responses"])
            }
            
            # Insert aggregation record
            aggregation_result = self.db.aggregations.insert_one(aggregation)
            aggregation_id = aggregation_result.inserted_id
            
            # Store company info
            if self.data["company_info"]:
                company_info = self.data["company_info"].copy()
                company_info["timestamp"] = datetime.datetime.utcnow()
                
                # Use upsert to update if exists or insert if not
                self.db.company_info.update_one(
                    {"name": company_info["name"]},
                    {"$set": company_info},
                    upsert=True
                )
            
            # Prepare content items for batch insert
            content_items = []
            
            # Process all platform data
            for platform, items in self.data["platforms"].items():
                for item in items:
                    content_item = item.copy()
                    content_item["aggregation_id"] = aggregation_id
                    
                    # Convert string dates to datetime objects
                    if platform == "spotify" and "release_date" in content_item:
                        content_item["created_at"] = datetime.datetime.fromisoformat(content_item["release_date"])
                    elif platform == "twitter" and "created_at" in content_item:
                        content_item["created_at"] = datetime.datetime.fromisoformat(content_item["created_at"].replace('Z', '+00:00'))
                    elif platform == "reddit" and "created_utc" in content_item:
                        content_item["created_at"] = datetime.datetime.fromisoformat(content_item["created_utc"])
                    elif platform == "youtube" and "publish_time" in content_item:
                        content_item["created_at"] = datetime.datetime.fromisoformat(content_item["publish_time"].replace('Z', '+00:00'))
                    elif platform == "news" and "publish_time" in content_item:
                        content_item["created_at"] = datetime.datetime.fromisoformat(content_item["publish_time"])
                    else:
                        content_item["created_at"] = datetime.datetime.utcnow()
                    
                    content_items.append(content_item)
            
            # Insert all content items in a batch
            if content_items:
                self.db.content_items.insert_many(content_items)
            
            print(f"✅ Data stored in MongoDB: {len(content_items)} items under aggregation ID {aggregation_id}")
            return True
        except Exception as e:
            print(f"❌ Error storing data in MongoDB: {str(e)}")
            return False

    # Main method to fetch all data
    def fetch_all_data(self):
        start_time = time.time()
        print(f"🔍 Starting data collection for {self.company_name}" + 
              (f" and {self.person_name}" if self.person_name else ""))
        
        # First get company info
        self.fetch_company_info()
        
        # Then get social media data
        self.fetch_spotify_data()
        self.fetch_twitter_data()
        self.fetch_reddit_data()
        self.fetch_youtube_data()
        
        end_time = time.time()
        print(f"\n✅ Data collection completed in {(end_time - start_time):.2f} seconds")
        print(f"📊 Total items collected: {len(self.data['all_responses'])}")
        
        return self.data

    # Generate insights from the collected data
    def generate_insights(self):
        """Generate insights from the collected data"""
        insights = {
            "company": self.company_name,
            "person": self.person_name,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "sentiment_summary": {
                "overall": {"positive": 0, "neutral": 0, "negative": 0},
                "by_platform": {}
            },
            "top_topics": [],
            "top_keywords": {},
            "platform_distribution": {},
            "company_summary": {}
        }
        
        # Count items by platform
        for platform, items in self.data["platforms"].items():
            insights["platform_distribution"][platform] = len(items)
        
        # Initialize sentiment counters for each platform
        for platform in self.data["platforms"].keys():
            insights["sentiment_summary"]["by_platform"][platform] = {
                "positive": 0, "neutral": 0, "negative": 0
            }
        
        # Collect all keywords and sentiments
        all_keywords = {}
        topic_counts = {}
        all_entities = {}
        
        for platform, items in self.data["platforms"].items():
            for item in items:
                # Count sentiments
                if "sentiment" in item and "sentiment" in item["sentiment"]:
                    sentiment_value = item["sentiment"]["sentiment"]
                    insights["sentiment_summary"]["overall"][sentiment_value] += 1
                    insights["sentiment_summary"]["by_platform"][platform][sentiment_value] += 1
                
                # Collect keywords
                if "context" in item and "keywords" in item["context"]:
                    for keyword in item["context"]["keywords"]:
                        if keyword not in all_keywords:
                            all_keywords[keyword] = 0
                        all_keywords[keyword] += 1
                
                # Collect topics
                if "context" in item and "topics" in item["context"]:
                    for topic_list in item["context"]["topics"]:
                        for word in topic_list:
                            if word not in topic_counts:
                                topic_counts[word] = 0
                            topic_counts[word] += 1
                
                # Collect entities
                if "main_entities" in item:
                    for entity in item["main_entities"]:
                        if entity not in all_entities:
                            all_entities[entity] = 0
                        all_entities[entity] += 1
        
        # Get top keywords
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        insights["top_keywords"] = dict(sorted_keywords[:20])  # Top 20 keywords
        
        # Get top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        insights["top_topics"] = [topic for topic, count in sorted_topics[:10]]  # Top 10 topics
        
        # Get top entities
        sorted_entities = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)
        insights["top_entities"] = dict(sorted_entities[:15])  # Top 15 entities
        
        # Add company summary
        if self.data["company_info"]:
            company = self.data["company_info"]
            insights["company_summary"] = {
                "name": company.get("name", self.company_name),
                "sector": company.get("sector", ""),
                "industry": company.get("industry", ""),
                "description": company.get("description", "")[:200] + "..." if len(company.get("description", "")) > 200 else company.get("description", "")
            }
        
        return insights

# Example usage with command line arguments
def main():
    parser = argparse.ArgumentParser(description="Social Media Finance Aggregator")
    parser.add_argument("--company", required=True, help="Company name or ticker symbol")
    parser.add_argument("--person", help="Optional person name to include in search")
    parser.add_argument("--output", help="Output JSON filename")
    parser.add_argument("--no-mongodb", action="store_true", help="Skip MongoDB storage")
    
    args = parser.parse_args()
    
    # Create aggregator with company and optional person
    aggregator = SocialMediaAggregator(
        args.company, 
        args.person,
        use_mongodb=not args.no_mongodb
    )
    
    # Fetch all data
    result = aggregator.fetch_all_data()
    
    # Generate insights
    insights = aggregator.generate_insights()
    
    # Export to JSON
    if args.output:
        aggregator.export_to_json(args.output)
    else:
        aggregator.export_to_json()
    
    # Store in MongoDB if enabled
    if not args.no_mongodb:
        aggregator.store_in_mongodb()
    
    # Print insights summary
    print("\n📊 INSIGHTS SUMMARY:")
    print(f"Company: {insights['company']}")
    if insights['person']:
        print(f"Person: {insights['person']}")
    
    print("\nSentiment Distribution:")
    for sentiment, count in insights["sentiment_summary"]["overall"].items():
        print(f"  {sentiment.capitalize()}: {count}")
    
    print("\nTop Keywords:")
    for keyword, count in list(insights["top_keywords"].items())[:5]:
        print(f"  {keyword}: {count}")
    
    print("\nTop Topics:")
    for topic in insights["top_topics"][:5]:
        print(f"  {topic}")
    
    print("\nTop Entities:")
    for entity, count in list(insights.get("top_entities", {}).items())[:5]:
        print(f"  {entity}: {count}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
