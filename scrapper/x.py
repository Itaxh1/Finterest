import tweepy
import pandas as pd

# Replace with your own Bearer Token
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAACav0QEAAAAA4ZoIDmtFCFB1b1IM4Nz6P6p6hqg%3DDQHs71P3VRWVwfrmma6lEVeqm83ASL79BJocsrwy8lam7oBd76'

# Setup client
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

# Query
query = 'TSLA OR AAPL OR #StockMarket OR #Earnings lang:en -is:retweet'

# Collect Tweets
tweets = []
response = client.search_recent_tweets(query=query, tweet_fields=['created_at', 'text', 'author_id'], max_results=100)

if response.data:
    for tweet in response.data:
        tweets.append([tweet.id, tweet.text, tweet.created_at, tweet.author_id])

    # Save to CSV
    df = pd.DataFrame(tweets, columns=['id', 'text', 'created_at', 'author_id'])
    df.to_csv("twitter_finance_trends.csv", index=False)
    print("✅ Saved to twitter_finance_trends.csv")
else:
    print("⚠️ No data found!")
