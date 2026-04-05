"""
Download MovieLens 25M and TMDB metadata.

Usage:
    python scripts/download_data.py

Downloads:
    data/raw/ml-25m/         — MovieLens 25M ratings, movies, tags
    data/raw/tmdb/           — TMDB movie metadata (posters, plots)
                               requires TMDB_API_KEY in .env
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_ZIP = "data/raw/ml-25m.zip"
ML_DIR = "data/raw/ml-25m"


def download_file(url: str, dest: str):
    """Download with progress bar."""
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=Path(dest).name,
        total=total,
        unit="B",
        unit_scale=True,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_movielens():
    if Path(ML_DIR).exists():
        print(f"MovieLens already exists at {ML_DIR}, skipping.")
        return
    print("Downloading MovieLens 25M (~250MB)...")
    download_file(ML_URL, ML_ZIP)
    print("Extracting...")
    with zipfile.ZipFile(ML_ZIP, "r") as z:
        z.extractall("data/raw/")
    os.remove(ML_ZIP)
    print(f"✓ MovieLens 25M extracted to {ML_DIR}/")
    print("  Files:", [f.name for f in Path(ML_DIR).iterdir()])


def download_tmdb_metadata(movie_ids: list[int], out_dir: str = "data/raw/tmdb"):
    """
    Fetch TMDB metadata for a list of MovieLens movie IDs.
    Requires TMDB_API_KEY in .env

    We fetch: title, overview (plot), poster_path, genres, release_date.
    Poster images are downloaded separately for VLM cold start.
    """
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        print("TMDB_API_KEY not set in .env — skipping TMDB download.")
        print("Add TMDB_API_KEY=your_key to .env and re-run to fetch posters + plots.")
        return

    import json
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metadata = {}

    print(f"Fetching TMDB metadata for {len(movie_ids)} movies...")
    for movie_id in tqdm(movie_ids[:100]):  # cap at 100 for now, remove cap later
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            metadata[movie_id] = {
                "title": data.get("title"),
                "overview": data.get("overview"),
                "genres": [g["name"] for g in data.get("genres", [])],
                "release_date": data.get("release_date"),
                "poster_path": data.get("poster_path"),
            }

    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ TMDB metadata saved to {out_dir}/metadata.json")


if __name__ == "__main__":
    download_movielens()
    # TMDB: run after preprocessing to get the filtered movie ID list
    # download_tmdb_metadata(movie_ids=[...])
