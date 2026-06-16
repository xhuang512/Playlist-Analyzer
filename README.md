# Playlist Analyzer

A Spotify playlist analysis project that combines API data extraction, EDA, statistical inference, recommendation modelling, and a Flask web dashboard.

## Project Structure

```text
notebooks/
  1-Data Extraction Pipeline.ipynb
  2-EDA.ipynb
  3-Statistical Inference and Regression.ipynb
  4-Recommendation Modelling.ipynb
  data/
    df.csv
    candidate_tracks.csv
    recommendations.csv

web_app/
  app.py
  data_utils.py
  charts.py
  stats_summary.py
  templates/
  static/
```

## Run the Website

From the project root:

```bash
/opt/miniconda3/envs/cpsc330/bin/python web_app/app.py
```

Open:

```text
http://127.0.0.1:5001
```

The app reads the current exported CSV files from `notebooks/data/`.

