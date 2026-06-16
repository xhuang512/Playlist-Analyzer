from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


class SpotifyClientError(RuntimeError):
    pass


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_playlist_id(value: str) -> str:
    text = value.strip()
    if not text:
        raise SpotifyClientError("Please enter a Spotify playlist link or playlist ID.")

    if text.startswith("spotify:playlist:"):
        return text.split(":")[-1]

    if "open.spotify.com" in text:
        parsed = urlparse(text)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "playlist":
            return parts[1]

    if re.fullmatch(r"[A-Za-z0-9]{15,}", text):
        return text

    raise SpotifyClientError("That does not look like a valid Spotify playlist link.")


def get_access_token() -> str:
    load_env_file()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SpotifyClientError("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in .env.")

    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=20,
    )
    if response.status_code != 200:
        raise SpotifyClientError(f"Could not authenticate with Spotify: {response.status_code}")
    return response.json()["access_token"]


def spotify_get(url: str, token: str, params: dict | None = None) -> dict:
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=20,
    )
    if response.status_code == 404:
        raise SpotifyClientError("Playlist not found. Make sure the playlist is public or accessible.")
    if response.status_code != 200:
        raise SpotifyClientError(f"Spotify API request failed: {response.status_code}")
    return response.json()


def get_playlist_items(playlist_id: str, token: str) -> list[dict]:
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/items"
    params = {
        "limit": 100,
        "fields": "items(added_at,is_local,track(id,name,artists(id,name),album(id,name,release_date,images,external_urls),duration_ms,popularity,explicit,external_urls)),next",
    }
    items: list[dict] = []
    while url:
        data = spotify_get(url, token, params=params)
        items.extend(data.get("items", []))
        url = data.get("next")
        params = None
    return items


def playlist_items_to_df(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        track = item.get("track")
        if not track or not track.get("id"):
            continue

        artists = track.get("artists") or []
        artist_names = [artist.get("name") for artist in artists if artist.get("name")]
        artist_ids = [artist.get("id") for artist in artists if artist.get("id")]
        album = track.get("album") or {}
        album_images = album.get("images") or []
        external_urls = track.get("external_urls") or {}

        rows.append(
            {
                "track_id": track.get("id"),
                "track_name": track.get("name"),
                "artist_names": ", ".join(artist_names),
                "artist_ids": ", ".join(artist_ids),
                "main_artist_name": artist_names[0] if artist_names else None,
                "main_artist_id": artist_ids[0] if artist_ids else None,
                "album_id": album.get("id"),
                "album_name": album.get("name"),
                "album_image_url": album_images[0]["url"] if album_images else None,
                "album_url": (album.get("external_urls") or {}).get("spotify"),
                "release_date": album.get("release_date"),
                "duration_ms": track.get("duration_ms"),
                "duration_min": (track.get("duration_ms") or 0) / 60000,
                "popularity": track.get("popularity"),
                "explicit": track.get("explicit"),
                "track_url": external_urls.get("spotify"),
                "added_at": item.get("added_at"),
                "is_local": item.get("is_local"),
            }
        )
    return pd.DataFrame(rows)


def get_artist_metadata(df: pd.DataFrame, token: str, sleep_time: float = 0.05) -> pd.DataFrame:
    artist_ids = df["main_artist_id"].dropna().drop_duplicates().tolist()
    rows = []
    for start in range(0, len(artist_ids), 50):
        batch = artist_ids[start : start + 50]
        data = spotify_get(
            "https://api.spotify.com/v1/artists",
            token,
            params={"ids": ",".join(batch)},
        )
        for artist in data.get("artists", []):
            if not artist:
                continue
            rows.append(
                {
                    "main_artist_id": artist.get("id"),
                    "artist_name_from_api": artist.get("name"),
                    "main_artist_genres": artist.get("genres") or [],
                    "artist_popularity": artist.get("popularity"),
                    "artist_followers": (artist.get("followers") or {}).get("total"),
                    "artist_image_url": (artist.get("images") or [{}])[0].get("url"),
                    "artist_url": (artist.get("external_urls") or {}).get("spotify"),
                }
            )
        time.sleep(sleep_time)
    return pd.DataFrame(rows)


def detect_title_script(title: str) -> str:
    if not isinstance(title, str) or title.strip() == "":
        return "unknown"
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", title))
    has_japanese = bool(re.search(r"[\u3040-\u30ff]", title))
    has_korean = bool(re.search(r"[\uac00-\ud7af]", title))
    has_latin = bool(re.search(r"[A-Za-z]", title))
    scripts = []
    if has_chinese:
        scripts.append("chinese")
    if has_japanese:
        scripts.append("japanese")
    if has_korean:
        scripts.append("korean")
    if has_latin:
        scripts.append("latin")
    if len(scripts) == 0:
        return "unknown"
    if len(scripts) == 1:
        return scripts[0]
    return "mixed"


def map_title_script_to_label(script: str) -> str:
    mapping = {
        "chinese": "Chinese-script title",
        "japanese": "Japanese-script title",
        "korean": "Korean-script title",
        "latin": "Latin-script title",
        "mixed": "Mixed-script title",
        "unknown": "Unknown-script title",
    }
    return mapping.get(script, "Unknown-script title")


def get_language_tags_from_genres_and_title(row) -> list[str]:
    genres = row["main_artist_genres"]
    title_script = row["title_script"]
    tags = []

    genre_text = " ".join(genres).lower() if isinstance(genres, list) else str(genres).lower()
    chinese_keywords = ["mandopop", "c-pop", "cantopop", "taiwanese pop", "chinese"]
    korean_keywords = ["k-pop", "korean"]
    japanese_keywords = ["j-pop", "japanese", "anime"]
    western_keywords = [
        "rock",
        "metal",
        "alternative",
        "indie",
        "punk",
        "folk",
        "soul",
        "country",
        "edm",
        "electronic",
        "dance",
        "hip hop",
        "rap",
        "r&b",
        "pop",
    ]

    if any(word in genre_text for word in chinese_keywords) or title_script == "chinese":
        tags.append("Chinese-associated")
    if any(word in genre_text for word in korean_keywords) or title_script == "korean":
        tags.append("Korean-associated")
    if any(word in genre_text for word in japanese_keywords) or title_script == "japanese":
        tags.append("Japanese-associated")
    if not tags and (any(word in genre_text for word in western_keywords) or title_script == "latin"):
        tags.append("Western-associated")
    if not tags:
        tags.append("Unknown/Other")
    return tags


def fetch_playlist_dataframe(playlist_link: str) -> pd.DataFrame:
    playlist_id = parse_playlist_id(playlist_link)
    token = get_access_token()
    items = get_playlist_items(playlist_id, token)
    df = playlist_items_to_df(items)
    if df.empty:
        raise SpotifyClientError("No playable tracks were found in this playlist.")

    artist_df = get_artist_metadata(df, token)
    if artist_df.empty:
        raise SpotifyClientError("Could not fetch artist metadata for this playlist.")

    df = df.merge(artist_df, on="main_artist_id", how="left")
    df["title_script"] = df["track_name"].apply(detect_title_script)
    df["song_title_language"] = df["title_script"].apply(map_title_script_to_label)
    df["genre_language_tags"] = df.apply(get_language_tags_from_genres_and_title, axis=1)
    return df
