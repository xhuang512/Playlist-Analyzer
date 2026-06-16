from __future__ import annotations

from flask import Flask, render_template, request

from charts import all_chart_specs
from data_utils import (
    artist_ranking_items,
    language_visual_summary,
    playlist_signal_summary,
    playlist_summary,
    popularity_language_summary,
    prepare_playlist,
)
from spotify_client import SpotifyClientError, fetch_playlist_dataframe
from stats_summary import inference_summary


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    playlist_link = ""
    error = None
    summary = None
    signals = None
    artist_items = []
    language_visual = None
    popularity_language = []
    charts = {}
    stats = None
    cover_urls = []
    cover_items = []
    has_results = False

    if request.method == "POST":
        playlist_link = request.form.get("playlist_link", "").strip()
        try:
            raw_playlist = fetch_playlist_dataframe(playlist_link)
            playlist = prepare_playlist(raw_playlist)
            summary = playlist_summary(playlist)
            signals = playlist_signal_summary(playlist)
            artist_items = artist_ranking_items(playlist)
            language_visual = language_visual_summary(playlist)
            popularity_language = popularity_language_summary(playlist)
            charts = all_chart_specs(playlist)
            stats = inference_summary(playlist)
            if "album_image_url" in playlist:
                cover_urls = (
                    playlist["album_image_url"]
                    .dropna()
                    .drop_duplicates()
                    .head(42)
                    .tolist()
                )
                album_group_cols = [
                    "album_id",
                    "album_name",
                    "album_image_url",
                    "album_url",
                    "release_date",
                    "main_artist_name",
                ]
                available_album_cols = [col for col in album_group_cols if col in playlist.columns]
                album_source = playlist.dropna(subset=["album_image_url"]).copy()
                if "album_url" not in album_source:
                    album_source["album_url"] = ""
                album_cards = (
                    album_source.groupby(available_album_cols, dropna=False)
                    .agg(
                        tracks=("track_name", lambda values: list(dict.fromkeys(values))[:4]),
                        track_count=("track_id", "count"),
                        max_popularity=("popularity", "max"),
                    )
                    .reset_index()
                    .sort_values(["max_popularity", "track_count"], ascending=False)
                    .head(42)
                )
                album_cards["release_date"] = album_cards["release_date"].dt.strftime("%Y-%m-%d").fillna("Unknown")
                album_cards["track_preview"] = album_cards["tracks"].apply(lambda values: " | ".join(values))
                cover_items = album_cards.to_dict(orient="records")
            has_results = True
        except (SpotifyClientError, ValueError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        playlist_link=playlist_link,
        error=error,
        has_results=has_results,
        summary=summary,
        signals=signals,
        artist_items=artist_items,
        language_visual=language_visual,
        popularity_language=popularity_language,
        charts=charts,
        stats=stats,
        cover_urls=cover_urls,
        cover_items=cover_items,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
