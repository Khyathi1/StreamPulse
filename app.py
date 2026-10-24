"""
Streams vs. Screens: Streamlit dashboard (CSCE 5320 project)

Run locally:   streamlit run app.py
Put Spotify_Youtube.csv in the data/ folder (or upload it in the sidebar).

Requires: streamlit pandas numpy scipy scikit-learn plotly
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Streams vs. Screens", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(BASE_DIR, "data", "Spotify_Youtube.csv")
FEATURES = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms"]
GROUP_ORDER = ["Spotify-skewed", "On trend", "YouTube-skewed", "Hit on both"]


# ----------------------------------------------------------------------------
# Data loading, cleaning, feature engineering
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(source):
    """source is either a file path (str) or the bytes of an uploaded file."""
    info = {}
    import io
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        df = pd.read_csv(handle)
        info["parser_warning"] = None
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        if isinstance(source, bytes):
            handle = io.BytesIO(source)
        df = pd.read_csv(handle, engine="python", on_bad_lines="skip",
                         encoding_errors="replace")
        info["parser_warning"] = f"Standard CSV read failed ({e}); malformed rows were skipped."
    info["rows_loaded"] = len(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    info["missing"] = {c: int(df[c].isna().sum())
                       for c in ["Stream", "Views", "Likes", "Comments"] if c in df.columns}

    n = len(df)
    df = df.drop_duplicates(subset="Uri" if "Uri" in df.columns else ["Artist", "Track"])
    info["duplicates_removed"] = n - len(df)
    n = len(df)
    df = df.dropna(subset=["Stream", "Views"])
    info["dropped_missing"] = n - len(df)
    n = len(df)
    df = df[(df["Stream"] > 0) & (df["Views"] > 0)]
    info["dropped_zero"] = n - len(df)
    df = df.dropna(subset=[f for f in FEATURES if f in df.columns])
    info["final_n"] = len(df)

    df["log_streams"] = np.log10(df["Stream"])
    df["log_views"] = np.log10(df["Views"])
    df["hit_spotify"] = (df["Stream"] >= df["Stream"].quantile(0.75)).astype(int)
    df["hit_youtube"] = (df["Views"] >= df["Views"].quantile(0.75)).astype(int)
    df["hit_both"] = df["hit_spotify"] & df["hit_youtube"]
    slope, intercept = np.polyfit(df["log_streams"], df["log_views"], 1)
    df["resid"] = df["log_views"] - (slope * df["log_streams"] + intercept)
    lo, hi = df["resid"].quantile(0.10), df["resid"].quantile(0.90)
    df["group"] = "On trend"
    df.loc[df["resid"] >= hi, "group"] = "YouTube-skewed"
    df.loc[df["resid"] <= lo, "group"] = "Spotify-skewed"
    df.loc[df["hit_both"] == 1, "group"] = "Hit on both"
    info["fit"] = (slope, intercept)
    if "official_video" in df.columns:
        df["official_video"] = df["official_video"].map(
            {True: "Official", False: "Not official"}).fillna("Unknown")
    if "Album_type" in df.columns:
        df["Album_type"] = df["Album_type"].fillna("unknown")
    return df.reset_index(drop=True), info


def corr_row(g):
    r, _ = stats.pearsonr(g["log_streams"], g["log_views"])
    rho, _ = stats.spearmanr(g["log_streams"], g["log_views"])
    return r, rho


# Plain plotly helpers that avoid px's internal groupby (it breaks on some
# pandas/plotly version combinations with a KeyError).
def grouped_bar(d, x, y, color, height=None, title=None, yrange=None):
    fig = go.Figure()
    for c in pd.unique(d[color]):
        m = (d[color] == c).to_numpy()
        fig.add_bar(x=d.loc[m, x].astype(str).tolist(), y=d.loc[m, y].tolist(), name=str(c))
    fig.update_layout(barmode="group", height=height, title=title,
                      xaxis_title=x, yaxis_title=y)
    if yrange:
        fig.update_yaxes(range=yrange)
    return fig


def group_box(d, y, by, order, title=None, height=360):
    fig = go.Figure()
    for g in order:
        vals = d.loc[(d[by] == g).to_numpy(), y].tolist()
        if vals:
            fig.add_trace(go.Box(y=vals, name=g, boxpoints=False))
    fig.update_layout(title=title, height=height, showlegend=False, yaxis_title=y)
    return fig


# ----------------------------------------------------------------------------
# Sidebar: data source and filters
# ----------------------------------------------------------------------------
st.title("Streams vs. Screens")
st.caption("Cross-platform popularity on Spotify and YouTube, and the limits of "
           "audio-feature-driven hit prediction. CSCE 5320 project dashboard.")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload Spotify_Youtube.csv", type="csv", key="csv_upload")
    path = st.text_input("...or file path", DEFAULT_PATH)

if upload is not None:
    source = upload.getvalue()
elif os.path.exists(path):
    source = path
else:
    st.info("Upload the CSV in the sidebar, or enter a valid file path.")
    st.stop()

full, info = load_data(source)
if info["parser_warning"]:
    st.warning(info["parser_warning"])
if info["rows_loaded"] < 19000:
    st.warning(f"Only {info['rows_loaded']:,} rows loaded; about 20,700 are expected. "
               "The file may be truncated. Re-download it before trusting the results.")

with st.sidebar:
    st.header("Filters")
    types = sorted(full["Album_type"].unique()) if "Album_type" in full else []
    sel_types = st.multiselect("Album type", types, default=types)
    vids = sorted(full["official_video"].unique()) if "official_video" in full else []
    sel_vids = st.multiselect("Video status", vids, default=vids)
    color_by = st.selectbox("Color scatter by", FEATURES + ["Album_type", "group"], index=0)
    sample_n = st.slider("Max points drawn in scatter", 1000, len(full), min(len(full), 20000),
                         step=1000) if len(full) > 1000 else len(full)

df = full
if types:
    df = df[df["Album_type"].isin(sel_types)]
if vids:
    df = df[df["official_video"].isin(sel_vids)]
if len(df) < 30:
    st.error("Fewer than 30 tracks match the filters. Loosen them in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "RQ1: Streams vs. views", "RQ2: Audio features",
     "RQ3: Over-performers", "RQ4: Hit prediction", "Data"])

with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows loaded", f"{info['rows_loaded']:,}")
    c2.metric("After cleaning", f"{info['final_n']:,}")
    c3.metric("Shown with filters", f"{len(df):,}")
    r, rho = corr_row(df)
    c4.metric("Spearman rho (streams vs views)", f"{rho:.3f}")
    st.subheader("Cleaning summary")
    st.write(f"- Duplicates removed: **{info['duplicates_removed']}**")
    st.write(f"- Dropped for missing streams/views: **{info['dropped_missing']}**")
    st.write(f"- Dropped for zero streams/views: **{info['dropped_zero']}**")
    st.write("- Missing values before cleaning: " +
             ", ".join(f"{k} = {v}" for k, v in info["missing"].items()))
    st.write(f"- Log-log trend: log10(views) = {info['fit'][0]:.3f} x log10(streams) "
             f"+ {info['fit'][1]:.3f}")

with tab1:
    st.subheader("Do popular tracks on Spotify tend to be popular on YouTube?")
    plot_df = df.sample(min(sample_n, len(df)), random_state=1)
    hover = [c for c in ["Artist", "Track", "Album_type"] if c in df]
    axis_labels = {"log_streams": "Spotify streams (log10)", "log_views": "YouTube views (log10)"}
    if color_by in FEATURES:
        fig = px.scatter(plot_df, x="log_streams", y="log_views", color=color_by,
                         hover_data=hover, opacity=0.5, render_mode="webgl", height=560,
                         color_continuous_scale="Viridis", labels=axis_labels)
    else:
        fig = go.Figure()
        for c in pd.unique(plot_df[color_by]):
            sub = plot_df[(plot_df[color_by] == c).to_numpy()]
            fig.add_trace(go.Scattergl(x=sub["log_streams"], y=sub["log_views"], mode="markers",
                                       name=str(c), opacity=0.5, marker=dict(size=5),
                                       text=sub["Artist"] + " - " + sub["Track"]
                                       if {"Artist", "Track"} <= set(sub.columns) else None))
        fig.update_layout(height=560, xaxis_title=axis_labels["log_streams"],
                          yaxis_title=axis_labels["log_views"])
    slope, intercept = info["fit"]
    xs = np.array([df["log_streams"].min(), df["log_streams"].max()])
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines",
                    line=dict(color="red", dash="dash"), name="Linear trend")
    st.plotly_chart(fig, width="stretch")

    rows = []
    r, rho = corr_row(df)
    rows.append(("All (filtered)", "all", len(df), r, rho))
    for col in ["Album_type", "official_video"]:
        if col in df:
            for val, g in df.groupby(col):
                if len(g) >= 30:
                    r, rho = corr_row(g)
                    rows.append((col, val, len(g), r, rho))
    ctab = pd.DataFrame(rows, columns=["grouping", "value", "n", "pearson_r", "spearman_rho"])
    st.markdown("**Correlation of log streams and log views, by group**")
    st.dataframe(ctab.round(3), width="stretch", hide_index=True)
    bars = ctab[ctab["grouping"] != "All (filtered)"].reset_index(drop=True)
    st.plotly_chart(grouped_bar(bars, "value", "spearman_rho", "grouping",
                                title="Spearman rho by group", yrange=[0, 1]), width="stretch")

with tab2:
    st.subheader("Which audio features go with popularity on each platform?")
    rows = []
    for f in FEATURES:
        rows.append((f, stats.spearmanr(df[f], df["log_streams"])[0],
                     stats.spearmanr(df[f], df["log_views"])[0]))
    c2t = pd.DataFrame(rows, columns=["feature", "Spotify streams", "YouTube views"])
    long = c2t.melt(id_vars="feature", var_name="platform", value_name="spearman_rho")
    st.plotly_chart(grouped_bar(long, "feature", "spearman_rho", "platform", height=480,
                                title="Spearman correlation of audio features with popularity"),
                    width="stretch")
    st.dataframe(c2t.round(3), width="stretch", hide_index=True)
    feat = st.selectbox("Look at one feature in detail", FEATURES)
    cA, cB = st.columns(2)
    cA.plotly_chart(px.density_heatmap(df, x=feat, y="log_streams", nbinsx=30, nbinsy=30,
                                       title=f"{feat} vs. streams"), width="stretch")
    cB.plotly_chart(px.density_heatmap(df, x=feat, y="log_views", nbinsx=30, nbinsy=30,
                                       title=f"{feat} vs. views"), width="stretch")

with tab3:
    st.subheader("What do tracks that over-perform on one platform look like?")
    st.caption("Spotify-skewed / YouTube-skewed = bottom / top 10% of residuals from the "
               "log-log trend. 'Hit on both' = top quartile on both platforms.")
    counts = df["group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    st.write("Group sizes: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    feats = st.multiselect("Features to compare", FEATURES,
                           default=["Danceability", "Energy", "Valence", "Acousticness"])
    for f in feats:
        st.plotly_chart(group_box(df, f, "group", GROUP_ORDER, title=f), width="stretch")
    means = df.groupby("group")[FEATURES].mean().reindex(GROUP_ORDER).round(3)
    st.markdown("**Mean audio features by group**")
    st.dataframe(means, width="stretch")
    if "Album_type" in df:
        share = (pd.crosstab(df["group"], df["Album_type"], normalize="index") * 100).round(1)
        st.markdown("**Album type share within each group (%)**")
        st.dataframe(share.reindex(GROUP_ORDER), width="stretch")

with tab4:
    st.subheader("How much of 'hit' status do audio features explain?")
    st.caption("Hit = top quartile of streams (or views). AUC of 0.5 means no better than chance. "
               "Models use the filtered data in the sidebar and 5-fold cross-validation.")
    if st.button("Train models"):
        X = df[FEATURES].values
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rows, imps = [], {}
        with st.spinner("Training..."):
            for target, label in [("hit_spotify", "Spotify hit"), ("hit_youtube", "YouTube hit")]:
                y = df[target].values
                logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                            random_state=42, n_jobs=-1)
                for name, model in [("Logistic regression", logit), ("Random forest", rf)]:
                    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                    rows.append((label, name, auc.mean(), acc.mean(), 1 - y.mean()))
                rf.fit(X, y)
                imps[label] = pd.Series(rf.feature_importances_, index=FEATURES)
        res = pd.DataFrame(rows, columns=["target", "model", "CV AUC", "CV accuracy",
                                          "majority-class accuracy"])
        st.dataframe(res.round(3), width="stretch", hide_index=True)
        cols = st.columns(2)
        for col, (label, imp) in zip(cols, imps.items()):
            col.plotly_chart(px.bar(imp.sort_values(), orientation="h",
                                    title=f"Random forest importance: {label}"),
                             width="stretch")
    else:
        st.info("Click 'Train models' to run the cross-validated comparison.")

with tab5:
    cols = [c for c in ["Artist", "Track", "Album_type", "Stream", "Views", "group"] + FEATURES
            if c in df]
    st.dataframe(df[cols], width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", df[cols].to_csv(index=False),
                       "filtered_tracks.csv", "text/csv")