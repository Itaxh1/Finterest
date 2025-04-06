from googleapiclient.discovery import build
import pandas as pd

api_key = "AIzaSyDal9-H_AOQSKWw3YuHBnF-oky2iAOJJZo"  # 🔑 Replace with your key

youtube = build('youtube', 'v3', developerKey=api_key)

search_query = "TSLA OR AAPL OR Stock Market OR Earnings"
results = []

request = youtube.search().list(
    q=search_query,
    part="snippet",
    type="video",
    maxResults=20,
    order="date"
)
response = request.execute()

for item in response["items"]:
    video_data = {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "publish_time": item["snippet"]["publishedAt"],
        "description": item["snippet"]["description"],
        "video_id": item["id"]["videoId"],
        "video_url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
    }
    results.append(video_data)

df = pd.DataFrame(results)
df.to_csv("youtube_finance_trends.csv", index=False)

print("✅ YouTube data saved to youtube_finance_trends.csv")
