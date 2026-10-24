# StreamPulse

**Cross-platform popularity on Spotify and YouTube, and the limits of audio-feature-driven hit prediction.**

StreamPulse compares how tracks perform on Spotify (streams) and YouTube (views), and tests how much a track's audio features (danceability, energy, loudness and so on) explain that popularity. It includes a reproducible analysis script and an interactive Streamlit dashboard.

Built for CSCE 5320 Scientific Data Visualization (University of North Texas).

## Research questions

1. **RQ1:** How well do Spotify streams and YouTube views correlate, and is the relationship the same across album types and video types?
2. **RQ2:** Which audio features correlate most with popularity on each platform, and do the platforms agree?
3. **RQ3:** What do tracks that over-perform on one platform have in common, compared with tracks that are popular on both?
4. **RQ4:** How much of "hit" status (top quartile) can audio features alone explain?

## Key findings

Results from 17,927 tracks after cleaning:

| Question | Result |
|---|---|
| RQ1 | Streams and views correlate at Spearman rho 0.61. Official videos 0.69 vs. non-official 0.30; singles 0.63, albums 0.62, compilations 0.38. |
| RQ2 | No audio feature correlates above 0.25 with either platform. Loudness is the strongest (0.13 with streams, 0.25 with views); instrumentalness is the most negative. Eight of ten features correlate more strongly with views. |
| RQ3 | YouTube-skewed tracks are louder, higher in energy, and more often singles (34%) than Spotify-skewed tracks (16%). Hits on both platforms are the loudest and least instrumental group. |
| RQ4 | Audio features alone predict hits weakly: cross-validated AUC 0.62 to 0.68, pseudo-R-squared 0.03 to 0.05. Accuracy equals the always-"not a hit" baseline (0.75). |

These are correlations from a single snapshot of popular tracks. The dataset has no release dates, playlist, or marketing variables.

## Dashboard

The Streamlit app has six tabs:

- **Overview:** cleaning summary and headline correlation
- **RQ1:** streams vs. views scatter and correlations by group
- **RQ2:** feature correlation bars and per-feature heatmaps
- **RQ3:** boxplots and tables for over-performing groups
- **RQ4:** cross-validated hit-prediction models and feature importances
- **Data:** filtered table with CSV download

Sidebar filters: album type, video status, scatter color, and maximum points drawn.

## Project structure

```
streampulse/
  app.py             Streamlit dashboard
  analysis.py        Full analysis; writes figures and summary.txt to results/
  requirements.txt   Python dependencies
  data/              Place Spotify_Youtube.csv here
  results/           Output of analysis.py (created automatically)
  .vscode/           Run and debug configurations
```

## Setup

Requires Python 3.10 or newer.

Windows (PowerShell):

```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `streamlit` is not recognized after installing, run it through Python with `py -m streamlit run app.py` (or `python -m streamlit run app.py`).

## Data

Download `Spotify_Youtube.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) (salvatorerastelli) and place it in `data/`. The file has 20,718 rows and 28 columns (about 30.8 MB). If the scripts warn that fewer than about 19,000 rows loaded, the file is damaged; download it again.

## Usage

Run the analysis (prints every number and saves figures and tables to `results/`):

```
python analysis.py
```

Run the dashboard:

```
streamlit run app.py
```

The dashboard reads `data/Spotify_Youtube.csv` by default, or you can upload the file in the sidebar. You can also use the Run and Debug panel in VS Code (F5).

### Outputs in `results/`

- `summary.txt`: every number used in the report
- `fig1_streams_vs_views`, `fig2_feature_correlations` (PNG and HTML), `fig3_group_comparison.png`, `fig4_rq4_importances.png`, `dashboard.html`
- `rq1_correlations.csv`, `rq2_feature_correlations.csv`, `rq3_group_feature_means.csv`, `rq4_model_results.csv`

## Method summary

- **Cleaning:** duplicates removed by Spotify URI (1,856), rows missing streams or views removed (934), one further row with missing audio features removed.
- **Features:** log10 of streams and views; hit labels at the top quartile of each platform (streams of at least 135,208,611; views of at least 67,507,614); over-performance groups from the residual of a log-log linear fit (bottom and top 10% are Spotify-skewed and YouTube-skewed).
- **Statistics:** Pearson and Spearman correlations, Kruskal-Wallis tests across groups.
- **Models:** logistic regression and random forest with 5-fold stratified cross-validation, reported with AUC, accuracy, and the majority-class baseline.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, Plotly, Matplotlib, Streamlit.

## Limitations

- The data is a single snapshot with no release dates, so popularity over time cannot be studied.
- The sample favors popular artists and tracks.
- Results are associations, not causes; playlist placement, artist fame, and marketing are not in the data.
- With about 18,000 tracks, nearly every difference is statistically significant, so effect sizes matter more than p-values.
