
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import tweepy
import praw
from googleapiclient.discovery import build
import pandas as pd

# -------- Spotify Podcast Scraper --------
print("\n🎧 Fetching Spotify Podcasts...")
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='fd1efa42cf584fb2b977e7db374470a7',
    client_secret='a1836b8f510e4efda2e471f3692897a0'))

podcast_results = sp.search(q="finance", type='show', limit=5)
episodes_data = []
for show in podcast_results['shows']['items']:
    show_id = show['id']
    show_name = show['name']
    episodes = sp.show_episodes(show_id, limit=10)
    for ep in episodes['items']:
        episodes_data.append({
            'podcast': show_name,
            'title': ep['name'],
            'description': ep['description'],
            'release_date': ep['release_date'],
            'audio': ep['audio_preview_url'],
            'url': ep['external_urls']['spotify']
        })
pd.DataFrame(episodes_data).to_csv("spotify_finance_podcasts.csv", index=False)
print("✅ Spotify data saved.")

# -------- Twitter Scraper --------
print("\n🐦 Fetching Twitter Tweets...")
client = tweepy.Client(
    bearer_token='AAAAAAAAAAAAAAAAAAAAACav0QEAAAAA4ZoIDmtFCFB1b1IM4Nz6P6p6hqg%3DDQHs71P3VRWVwfrmma6lEVeqm83ASL79BJocsrwy8lam7oBd76',
    wait_on_rate_limit=True)

twitter_query = 'TSLA OR AAPL OR #StockMarket OR #Earnings lang:en -is:retweet'
twitter_tweets = []
twitter_response = client.search_recent_tweets(query=twitter_query, tweet_fields=['created_at', 'text', 'author_id'], max_results=100)
if twitter_response.data:
    for tweet in twitter_response.data:
        twitter_tweets.append([tweet.id, tweet.text, tweet.created_at, tweet.author_id])
    pd.DataFrame(twitter_tweets, columns=['id', 'text', 'created_at', 'author_id']).to_csv("twitter_finance_trends.csv", index=False)
    print("✅ Twitter data saved.")
else:
    print("⚠️ No Twitter data found.")

# -------- Reddit Scraper --------
print("\n👽 Fetching Reddit Posts...")
reddit = praw.Reddit(
    client_id="I8h-Ft5F2LbnqkYj8RYBZQ",
    client_secret="_0m9SWFQH8QI2mPUUmla24bNCv1wQw",
    user_agent="finterest-script by u/TrainingEmphasis9633",
    username="TrainingEmphasis9633",
    password="Buntysan2012@")

keywords = ["TSLA", "AAPL", "stock market", "earnings"]
subreddits = ["stocks", "investing", "wallstreetbets"]
reddit_posts = []
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for submission in subreddit.new(limit=300):
        if any(k.lower() in submission.title.lower() for k in keywords):
            reddit_posts.append([sub, submission.title, submission.selftext, submission.created_utc])
pd.DataFrame(reddit_posts, columns=["subreddit", "title", "body", "timestamp"]).to_csv("reddit_finance_trends.csv", index=False)
print("✅ Reddit data saved.")

# -------- YouTube Scraper --------
print("\n📺 Fetching YouTube Videos...")
youtube = build('youtube', 'v3', developerKey="AIzaSyDal9-H_AOQSKWw3YuHBnF-oky2iAOJJZo")
youtube_query = "TSLA OR AAPL OR Stock Market OR Earnings"
request = youtube.search().list(q=youtube_query, part="snippet", type="video", maxResults=20, order="date")
response = request.execute()

youtube_results = []
for item in response["items"]:
    youtube_results.append({
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "publish_time": item["snippet"]["publishedAt"],
        "description": item["snippet"]["description"],
        "video_id": item["id"]["videoId"],
        "video_url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
    })
pd.DataFrame(youtube_results).to_csv("youtube_finance_trends.csv", index=False)
print("✅ YouTube data saved.")
