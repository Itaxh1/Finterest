import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

client_id = 'fd1efa42cf584fb2b977e7db374470a7'
client_secret = 'a1836b8f510e4efda2e471f3692897a0'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Search for finance podcasts
results = sp.search(q="finance", type='show', limit=5)

episodes_data = []

for show in results['shows']['items']:
    show_id = show['id']
    show_name = show['name']
    show_desc = show['description']
    
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

df = pd.DataFrame(episodes_data)
df.to_csv("spotify_finance_podcasts.csv", index=False)
print("✅ Saved top episodes to spotify_finance_podcasts.csv")
