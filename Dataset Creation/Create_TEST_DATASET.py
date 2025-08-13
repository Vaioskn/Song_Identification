import os
import csv
import time
import requests
import random

CLIENT_ID = '<REDUCTED FOR GITHUB>'
DATA_DIR = 'TEST_METADATA'
INPUT_CSV = os.path.join(DATA_DIR, 'metadata_1000.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'metadata_with_genre.csv')
AUDIO_DIR = os.path.join(DATA_DIR, 'downloaded_tracks')
API_URL = 'https://api.jamendo.com/v3.0/tracks'
MAX_SONGS_TO_GET = 200
ID_RANGE = (1, 10000)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

with open(INPUT_CSV, newline='', encoding='utf-8') as f:
    existing_ids = {row['track_id'] for row in csv.DictReader(f)}

fieldnames = ['track_id', 'title', 'artist', 'album', 'duration_sec',
              'download_url', 'filename', 'genre']
writer = csv.DictWriter(open(OUTPUT_CSV, 'w', newline='', encoding='utf-8'),
                        fieldnames=fieldnames)
writer.writeheader()

downloaded = 0
checked_ids = set()

while downloaded < MAX_SONGS_TO_GET:
    tid = str(random.randint(*ID_RANGE))
    if tid in existing_ids or tid in checked_ids:
        continue
    checked_ids.add(tid)

    params = {
        'client_id': CLIENT_ID,
        'id': tid,
        'format': 'json',
        'include': 'musicinfo',
        'limit': 1,
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get('results', [])
    except Exception:
        continue

    if not results:
        continue

    entry = results[0]
    musicinfo = entry.get('musicinfo', {})
    genres = musicinfo.get('tags', {}).get('genres', [])
    if not genres:
        continue

    genre = genres[0]
    metadata = {
        'track_id': entry['id'],
        'title': entry.get('name', ''),
        'artist': entry.get('artist_name', ''),
        'album': entry.get('album_name', ''),
        'duration_sec': entry.get('duration', ''),
        'download_url': entry.get('audiodownload', ''),
        'filename': f"{entry['id']}.mp3",
        'genre': genre
    }
    writer.writerow(metadata)

    download_url = metadata['download_url']
    try:
        audio_resp = requests.get(download_url, stream=True, timeout=20)
        audio_resp.raise_for_status()
        file_path = os.path.join(AUDIO_DIR, metadata['filename'])
        with open(file_path, 'wb') as af:
            for chunk in audio_resp.iter_content(1024):
                af.write(chunk)
    except Exception:
        continue

    downloaded += 1
    print(f"[{downloaded}/{MAX_SONGS_TO_GET}] Downloaded track ID {tid}: {metadata['title']} ({genre})")
    time.sleep(1.0)

print(f"Done. Retrieved {downloaded} new tracks. Metadata saved to {OUTPUT_CSV} and audio files in {AUDIO_DIR}")
