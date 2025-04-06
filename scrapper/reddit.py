import praw
import pandas as pd

reddit = praw.Reddit(
    client_id="I8h-Ft5F2LbnqkYj8RYBZQ",  # ✅ Use this!
    client_secret="_0m9SWFQH8QI2mPUUmla24bNCv1wQw",
    user_agent="finterest-script by u/TrainingEmphasis9633",
    username="TrainingEmphasis9633",
    password="Buntysan2012@"
)

keywords = ["TSLA", "AAPL", "stock market", "earnings"]
subreddits = ["stocks", "investing", "wallstreetbets"]

posts = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for submission in subreddit.new(limit=300):
        if any(k.lower() in submission.title.lower() for k in keywords):
            posts.append([sub, submission.title, submission.selftext, submission.created_utc])

df = pd.DataFrame(posts, columns=["subreddit", "title", "body", "timestamp"])
df.to_csv("reddit_finance_trends.csv", index=False)

print(reddit.user.me())
