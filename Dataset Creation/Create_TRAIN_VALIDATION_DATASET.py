import os
import random
import requests
import pandas as pd
from tqdm import tqdm
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, TALB

CLIENT_ID = "<REDUCTED_FOR_GITHUB>"
AUDIO_DIR = "audio_1000"
METADATA_CSV = os.path.join(AUDIO_DIR, "metadata.csv")
FAILED_LOG = os.path.join(AUDIO_DIR, "failed_ids.txt")

os.makedirs(AUDIO_DIR, exist_ok=True)

if not os.path.exists(METADATA_CSV):
    print("Getting tracks from Jamendo API...")
    pool, page, per_page = [], 1, 200
    while len(pool) < 5000:
        resp = requests.get(
            "https://api.jamendo.com/v3.0/tracks/",
            params={
                "client_id": CLIENT_ID,
                "format": "json",
                "limit": per_page,
                "offset": (page-1)*per_page,
                "audioformat": "mp32",
                "filter": "audiodownload_allowed=1",
                "include": "musicinfo"
            },
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
        if not data:
            break
        pool.extend(data)
        page += 1
        print(f"-> Collected {len(pool)} tracks so far...")

    if len(pool) < 1000:
        raise RuntimeError("Insufficient tracks available")

    sampled = random.sample(pool, 1000)

    md = pd.DataFrame(sampled)[[
        "id", "name", "artist_name", "album_name", "duration", "audiodownload"
    ]]
    md = md.rename(columns={
        "id": "track_id",
        "name": "title",
        "artist_name": "artist",
        "album_name": "album",
        "duration": "duration_sec",
        "audiodownload": "download_url"
    })
    md.to_csv(METADATA_CSV, index=False)
    print(f"metadata.csv created with {len(md)} tracks.")

else:
    print("metadata.csv found, resuming downloads.")
    md = pd.read_csv(METADATA_CSV, dtype={"track_id": str})

md["filename"] = md["track_id"] + ".mp3"

failed = []
for _, row in tqdm(md.iterrows(), total=len(md), desc="Downloading MP3s"):
    out_path = os.path.join(AUDIO_DIR, row["filename"])
    if os.path.exists(out_path):
        continue

    try:
        resp = requests.get(row["download_url"], stream=True, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as wf:
            for chunk in resp.iter_content(8192):
                wf.write(chunk)

        try:
            audio = EasyID3(out_path)
        except:
            id3 = ID3(out_path)
            id3.add(TIT2(encoding=3, text=row["title"]))
            id3.add(TPE1(encoding=3, text=row["artist"]))
            id3.add(TALB(encoding=3, text=row["album"]))
            id3.save()
            audio = EasyID3(out_path)

        audio["title"]  = row["title"]
        audio["artist"] = row["artist"]
        audio["album"]  = row["album"]
        audio.save()

    except Exception as e:
        failed.append(row["track_id"])
        tqdm.write(f"Failed download {row['track_id']}: {e}")

if failed:
    with open(FAILED_LOG, "w") as f:
        f.write("\n".join(failed))
    print(f"\n{len(failed)} tracks failed to download. See {FAILED_LOG}")

mismatches, corrupted, missing = [], [], []

print("\nValidating downloaded MP3 durations...")
for _, row in tqdm(md.iterrows(), total=len(md)):
    file_path = os.path.join(AUDIO_DIR, row["filename"])
    if not os.path.exists(file_path):
        missing.append(row["filename"])
        continue
    try:
        audio = MP3(file_path)
        actual_duration = audio.info.length
        expected_duration = row["duration_sec"]
        if abs(actual_duration - expected_duration) > 5:
            mismatches.append({
                "filename": row["filename"],
                "expected": expected_duration,
                "actual": round(actual_duration, 2)
            })
    except Exception as e:
        corrupted.append((row["filename"], str(e)))

print("\nSummary:")

if missing:
    print(f"\nMissing files ({len(missing)}):")
    for f in missing:
        print(f"  - {f}")

if corrupted:
    print(f"\nCorrupted files ({len(corrupted)}):")
    for fname, err in corrupted:
        print(f"  - {fname}: {err}")

if mismatches:
    print(f"\nDuration mismatches ({len(mismatches)}):")
    for m in mismatches:
        print(f"  - {m['filename']}: expected {m['expected']}s, actual {m['actual']}s")

if not (missing or corrupted or mismatches or failed):
    print("All 1000 tracks downloaded")
else:
    total_issues = len(missing) + len(corrupted) + len(mismatches) + len(failed)
    print(f"\nCompleted with {total_issues} issues.")

print("\ncompleted.")
