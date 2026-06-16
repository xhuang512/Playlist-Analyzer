from __future__ import annotations

import altair as alt
import pandas as pd

from data_utils import genre_long


alt.data_transformers.disable_max_rows()

PALETTE = ["#7bd88f", "#6ec6d9", "#f3c56b", "#d98290", "#9b90d9", "#86c2a5"]


def _spec(chart: alt.Chart) -> dict:
    themed = (
        chart.configure_view(stroke=None, fill="transparent")
        .configure_axis(
            labelColor="#d9ded8",
            titleColor="#a9b2aa",
            gridColor="rgba(255,255,255,0.08)",
            domainColor="rgba(255,255,255,0.18)",
            tickColor="rgba(255,255,255,0.18)",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            labelColor="#d9ded8",
            titleColor="#a9b2aa",
            orient="bottom",
        )
        .configure(background="transparent")
    )
    return themed.to_dict()


def language_distribution(df: pd.DataFrame) -> dict:
    data = (
        df["song_title_language"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("song_title_language")
        .reset_index(name="track_count")
    )
    data["pct"] = data["track_count"] / data["track_count"].sum() * 100

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("track_count:Q", title="Number of tracks"),
            y=alt.Y("song_title_language:N", sort="-x", title=None),
            color=alt.Color("song_title_language:N", scale=alt.Scale(range=PALETTE), legend=None),
            tooltip=[
                alt.Tooltip("song_title_language:N", title="Title language/script"),
                alt.Tooltip("track_count:Q", title="Tracks"),
                alt.Tooltip("pct:Q", title="Share", format=".1f"),
            ],
        )
        .properties(width="container", height=220)
    )
    return _spec(chart)


def popularity_by_language(df: pd.DataFrame) -> dict:
    chart = (
        alt.Chart(df)
        .mark_boxplot(size=45)
        .encode(
            x=alt.X("song_title_language:N", title=None),
            y=alt.Y("popularity:Q", title="Track popularity"),
            color=alt.Color("song_title_language:N", legend=None),
            tooltip=["song_title_language:N", "popularity:Q"],
        )
        .properties(width="container", height=320)
    )
    return _spec(chart)


def top_genres(df: pd.DataFrame) -> dict:
    data = (
        genre_long(df)["genre"]
        .value_counts()
        .head(18)
        .rename_axis("genre")
        .reset_index(name="track_count")
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X("track_count:Q", title="Number of tracks"),
            y=alt.Y("genre:N", sort="-x", title=None),
            color=alt.Color("track_count:Q", scale=alt.Scale(range=["#b9dbc1", "#244438"]), legend=None),
            tooltip=["genre:N", "track_count:Q"],
        )
        .properties(width="container", height=430)
    )
    return _spec(chart)


def top_artists(df: pd.DataFrame) -> dict:
    agg_kwargs = {
        "track_count": ("track_id", "count"),
        "avg_track_popularity": ("popularity", "mean"),
        "artist_popularity": ("artist_popularity", "max"),
        "artist_followers": ("artist_followers", "max"),
    }
    if "artist_image_url" in df.columns:
        agg_kwargs["artist_image_url"] = ("artist_image_url", "first")

    data = (
        df.groupby("main_artist_name")
        .agg(**agg_kwargs)
        .sort_values(["track_count", "avg_track_popularity"], ascending=False)
        .head(14)
        .reset_index()
    )
    if "artist_image_url" not in data.columns:
        data["artist_image_url"] = None

    max_count = max(float(data["track_count"].max()), 1.0)
    label_space = max(5.0, max_count * 0.55)
    data["zero"] = 0
    data["image_x"] = -label_space * 0.86
    data["label_x"] = -label_space * 0.72
    artist_order = data["main_artist_name"].tolist()

    base = alt.Chart(data).encode(
        y=alt.Y("main_artist_name:N", sort=artist_order, axis=None),
        tooltip=[
            "main_artist_name:N",
            "track_count:Q",
            alt.Tooltip("avg_track_popularity:Q", format=".1f"),
            "artist_popularity:Q",
            alt.Tooltip("artist_followers:Q", format=","),
        ],
    )

    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=20).encode(
        x=alt.X(
            "zero:Q",
            title="Number of tracks",
            scale=alt.Scale(domain=[-label_space, max_count * 1.16]),
        ),
        x2="track_count:Q",
        color=alt.Color(
            "avg_track_popularity:Q",
            title="Avg popularity",
            scale=alt.Scale(range=["#6ec6d9", "#f3c56b"]),
        ),
    )

    images = base.transform_filter(
        alt.datum.artist_image_url != None
    ).mark_image(width=34, height=34).encode(
        x=alt.X("image_x:Q", scale=alt.Scale(domain=[-label_space, max_count * 1.16]), axis=None),
        url="artist_image_url:N",
    )

    names = base.mark_text(align="left", baseline="middle", dx=0, color="#f4efe4", fontWeight="bold").encode(
        x=alt.X("label_x:Q", scale=alt.Scale(domain=[-label_space, max_count * 1.16]), axis=None),
        text="main_artist_name:N",
    )

    counts = base.mark_text(align="left", baseline="middle", dx=6, color="#d9ded8", fontWeight="bold").encode(
        x=alt.X("track_count:Q", scale=alt.Scale(domain=[-label_space, max_count * 1.16])),
        text="track_count:Q",
    )

    chart = (
        (bars + images + names + counts)
        .resolve_scale(x="shared")
        .properties(width="container", height=430)
    )
    return _spec(chart)


def release_decade_counts(df: pd.DataFrame) -> dict:
    data = (
        df.loc[df["release_decade"].ne("Unknown"), "release_decade"]
        .value_counts()
        .rename_axis("release_decade")
        .reset_index(name="track_count")
    )
    decade_order = sorted(data["release_decade"].unique())
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("release_decade:N", title="Release decade", sort=decade_order),
            y=alt.Y("track_count:Q", title="Number of tracks"),
            color=alt.Color("track_count:Q", scale=alt.Scale(range=["#b9dbc1", "#244438"]), legend=None),
            tooltip=["release_decade:N", "track_count:Q"],
        )
        .properties(width="container", height=260)
    )
    return _spec(chart)


def tracks_added_over_time(df: pd.DataFrame) -> dict:
    data = (
        df.dropna(subset=["added_at"])
        .set_index("added_at")
        .resample("D")
        .size()
        .reset_index(name="tracks_added")
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("added_at:T", title="Date added"),
            y=alt.Y("tracks_added:Q", title="Tracks added"),
            tooltip=["added_at:T", "tracks_added:Q"],
        )
        .properties(width="container", height=260)
    )
    return _spec(chart)


def all_chart_specs(df: pd.DataFrame) -> dict[str, dict]:
    return {
        "releaseDecade": release_decade_counts(df),
        "tracksAdded": tracks_added_over_time(df),
    }
