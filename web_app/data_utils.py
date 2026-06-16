from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "notebooks" / "data"
PLAYLIST_PATH = DATA_DIR / "df.csv"
RECOMMENDATIONS_PATH = DATA_DIR / "recommendations.csv"


def parse_list_like(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_playlist() -> pd.DataFrame:
    df = pd.read_csv(PLAYLIST_PATH)
    return prepare_playlist(df)


def prepare_playlist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["added_at"] = pd.to_datetime(df["added_at"], errors="coerce", utc=True)
    df["release_year"] = df["release_date"].dt.year
    df["release_decade"] = (
        (df["release_year"] // 10 * 10)
        .astype("Int64")
        .astype(str)
        .str.replace("<NA>", "Unknown", regex=False)
        + "s"
    )
    df.loc[df["release_decade"].eq("Unknowns"), "release_decade"] = "Unknown"
    df["log_artist_followers"] = np.log1p(df["artist_followers"])
    df["explicit_int"] = df["explicit"].astype(int)
    df["high_popularity"] = (df["popularity"] >= df["popularity"].median()).astype(int)

    language_counts = df["song_title_language"].fillna("Unknown").value_counts()
    common_languages = language_counts[language_counts >= 10].index
    df["language_group"] = df["song_title_language"].fillna("Unknown")
    df["language_model_group"] = np.where(
        df["language_group"].isin(common_languages),
        df["language_group"],
        "Other/Mixed",
    )
    return df


def load_recommendations() -> pd.DataFrame:
    if not RECOMMENDATIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RECOMMENDATIONS_PATH)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def genre_long(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[
            [
                "track_id",
                "track_name",
                "main_artist_name",
                "main_artist_genres",
                "popularity",
                "song_title_language",
            ]
        ]
        .assign(genre=lambda x: x["main_artist_genres"].apply(parse_list_like))
        .explode("genre")
        .dropna(subset=["genre"])
    )


def playlist_summary(df: pd.DataFrame) -> dict:
    dominant_language = df["song_title_language"].fillna("Unknown").mode()
    dominant_language_value = dominant_language.iloc[0].replace("-script title", "") if not dominant_language.empty else "Unknown"
    explicit_share = df["explicit"].mean() * 100 if "explicit" in df else 0

    stat_cards = [
        {"label": "Total Tracks", "value": f"{df['track_id'].nunique():,}", "icon": "music-2"},
        {"label": "Unique Artists", "value": f"{df['main_artist_id'].nunique():,}", "icon": "users"},
        {"label": "Albums", "value": f"{df['album_id'].nunique():,}", "icon": "disc-3"},
        {"label": "Avg Popularity", "value": f"{df['popularity'].mean():.1f}", "icon": "star"},
        {"label": "Avg Duration", "value": f"{df['duration_min'].mean():.1f} min", "icon": "clock-3"},
        {"label": "Explicit Share", "value": f"{explicit_share:.0f}%", "icon": "badge-alert"},
        {"label": "Top Title Script", "value": dominant_language_value, "icon": "languages"},
        {"label": "Release Span", "value": f"{int(df['release_year'].min())}-{int(df['release_year'].max())}", "icon": "calendar-range"},
    ]

    artist_image_col = "artist_image_url" if "artist_image_url" in df else None
    artist_summary = (
        df.groupby(["main_artist_id", "main_artist_name"], dropna=False)
        .agg(
            track_count=("track_id", "count"),
            artist_image_url=(artist_image_col, "first") if artist_image_col else ("track_id", lambda _: None),
            artist_popularity=("artist_popularity", "max"),
        )
        .reset_index()
        .sort_values(["track_count", "artist_popularity"], ascending=False)
    )
    top_artist = artist_summary.iloc[0].to_dict() if not artist_summary.empty else {}

    popular_track_cols = [
        "track_name",
        "main_artist_name",
        "album_name",
        "popularity",
        "track_url",
        "album_image_url",
    ]
    available_track_cols = [col for col in popular_track_cols if col in df.columns]
    top_track = (
        df.sort_values("popularity", ascending=False)
        .head(1)[available_track_cols]
        .to_dict(orient="records")
    )
    top_track = top_track[0] if top_track else {}

    album_group_cols = ["album_id", "album_name", "main_artist_name"]
    if "album_image_url" in df:
        album_group_cols.append("album_image_url")
    if "album_url" in df:
        album_group_cols.append("album_url")
    album_summary = (
        df.groupby(album_group_cols, dropna=False)
        .agg(track_count=("track_id", "count"), max_popularity=("popularity", "max"))
        .reset_index()
        .sort_values(["track_count", "max_popularity"], ascending=False)
    )
    top_album = album_summary.iloc[0].to_dict() if not album_summary.empty else {}

    return {
        "stat_cards": stat_cards,
        "highlights": [
            {
                "label": "Most Frequent Artist",
                "title": top_artist.get("main_artist_name", "Unknown"),
                "subtitle": f"{int(top_artist.get('track_count', 0))} tracks in playlist",
                "image_url": top_artist.get("artist_image_url"),
                "url": None,
                "icon": "mic-2",
            },
            {
                "label": "Most Popular Track",
                "title": top_track.get("track_name", "Unknown"),
                "subtitle": f"{top_track.get('main_artist_name', 'Unknown')} · Popularity {top_track.get('popularity', 'Unknown')}",
                "image_url": top_track.get("album_image_url"),
                "url": top_track.get("track_url"),
                "icon": "sparkles",
            },
            {
                "label": "Most Represented Album",
                "title": top_album.get("album_name", "Unknown"),
                "subtitle": f"{top_album.get('main_artist_name', 'Unknown')} · {int(top_album.get('track_count', 0))} tracks",
                "image_url": top_album.get("album_image_url"),
                "url": top_album.get("album_url"),
                "icon": "album",
            },
        ],
    }


def top_metric_cards(df: pd.DataFrame) -> list[dict[str, str]]:
    return playlist_summary(df)["stat_cards"]


def playlist_signal_summary(df: pd.DataFrame) -> dict:
    genre_counts = genre_long(df)["genre"].value_counts().head(14)
    max_genre_count = genre_counts.max() if not genre_counts.empty else 1
    genre_chips = [
        {
            "name": genre,
            "count": int(count),
            "weight": float(0.72 + (count / max_genre_count) * 0.9),
            "size": float(72 + (count / max_genre_count) * 118),
        }
        for genre, count in genre_counts.items()
    ]

    language_counts = df["song_title_language"].fillna("Unknown").value_counts()
    language_total = language_counts.sum() if not language_counts.empty else 1
    language_mix = [
        {
            "label": label.replace("-script title", ""),
            "count": int(count),
            "pct": round(count / language_total * 100, 1),
        }
        for label, count in language_counts.head(5).items()
    ]

    track_cols = [
        "track_name",
        "main_artist_name",
        "album_name",
        "popularity",
        "track_url",
        "album_image_url",
    ]
    available_track_cols = [col for col in track_cols if col in df.columns]
    mainstream = df.sort_values("popularity", ascending=False).head(1)[available_track_cols].to_dict(orient="records")
    hidden_pool = df[df["popularity"] > 0].copy()
    hidden = hidden_pool.sort_values("popularity", ascending=True).head(1)[available_track_cols].to_dict(orient="records")

    return {
        "genre_chips": genre_chips,
        "language_mix": language_mix,
        "spectrum": {
            "mainstream": mainstream[0] if mainstream else {},
            "hidden": hidden[0] if hidden else {},
        },
    }


def language_visual_summary(df: pd.DataFrame) -> dict:
    counts = df["song_title_language"].fillna("Unknown").value_counts()
    total = int(counts.sum()) if not counts.empty else 1
    palette = ["#7bd88f", "#6ec6d9", "#f3c56b", "#d98290", "#9b90d9", "#86c2a5"]
    offset = 0
    segments = []
    for i, (label, count) in enumerate(counts.items()):
        pct = count / total * 100
        segments.append(
            {
                "label": label.replace("-script title", ""),
                "full_label": label,
                "count": int(count),
                "pct": round(pct, 1),
                "color": palette[i % len(palette)],
                "dash": f"{pct:.3f} {100 - pct:.3f}",
                "offset": round(-offset, 3),
            }
        )
        offset += pct
    return {"total": total, "segments": segments}


def popularity_language_summary(df: pd.DataFrame) -> list[dict]:
    rows = []
    for language, group in df.groupby("song_title_language", dropna=False):
        popularity = group["popularity"].dropna()
        if popularity.empty:
            continue
        top_track = group.sort_values("popularity", ascending=False).iloc[0]
        rows.append(
            {
                "label": str(language).replace("-script title", ""),
                "full_label": str(language),
                "count": int(len(group)),
                "mean": round(float(popularity.mean()), 1),
                "median": round(float(popularity.median()), 1),
                "q1": round(float(popularity.quantile(0.25)), 1),
                "q3": round(float(popularity.quantile(0.75)), 1),
                "min": round(float(popularity.min()), 1),
                "max": round(float(popularity.max()), 1),
                "top_track": top_track.get("track_name", "Unknown"),
                "top_artist": top_track.get("main_artist_name", "Unknown"),
                "top_popularity": int(top_track.get("popularity", 0)),
            }
        )
    rows.sort(key=lambda item: item["mean"], reverse=True)
    return rows


def artist_ranking_items(df: pd.DataFrame) -> list[dict]:
    agg_kwargs = {
        "track_count": ("track_id", "count"),
        "avg_track_popularity": ("popularity", "mean"),
        "artist_popularity": ("artist_popularity", "max"),
        "artist_followers": ("artist_followers", "max"),
    }
    if "artist_image_url" in df.columns:
        agg_kwargs["artist_image_url"] = ("artist_image_url", "first")
    if "artist_url" in df.columns:
        agg_kwargs["artist_url"] = ("artist_url", "first")

    data = (
        df.groupby("main_artist_name", dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values(["track_count", "avg_track_popularity"], ascending=False)
        .head(16)
    )
    if data.empty:
        return []
    if "artist_image_url" not in data.columns:
        data["artist_image_url"] = None
    if "artist_url" not in data.columns:
        data["artist_url"] = None

    max_count = max(float(data["track_count"].max()), 1.0)
    max_followers = max(float(data["artist_followers"].fillna(0).max()), 1.0)
    items = []
    for _, row in data.iterrows():
        count = int(row["track_count"])
        count_ratio = count / max_count
        lightness = 72 - count_ratio * 34
        items.append(
            {
                "name": row["main_artist_name"],
                "track_count": count,
                "avg_track_popularity": round(float(row["avg_track_popularity"]), 1),
                "artist_popularity": int(row["artist_popularity"]) if pd.notna(row["artist_popularity"]) else 0,
                "artist_followers": int(row["artist_followers"]) if pd.notna(row["artist_followers"]) else 0,
                "artist_followers_label": f"{int(row['artist_followers']):,}" if pd.notna(row["artist_followers"]) else "Unknown",
                "artist_image_url": row["artist_image_url"],
                "artist_url": row["artist_url"],
                "bar_pct": round(12 + count_ratio * 88, 2),
                "follower_pct": round((float(row["artist_followers"] or 0) / max_followers) * 100, 2),
                "bar_color": f"hsl(145 42% {lightness:.1f}%)",
            }
        )
    return items
