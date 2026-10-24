"""
Streams vs. Screens: Spotify streams vs. YouTube views (CSCE 5320 project)

Usage (from the project folder):
    python analysis.py
    python analysis.py --csv data/Spotify_Youtube.csv --out results

Requires: pandas numpy scipy scikit-learn matplotlib plotly
Outputs (in --out):
    summary.txt                 every number needed for the report
    rq1_correlations.csv        streams vs views correlation, overall and by group
    rq2_feature_correlations.csv
    rq3_group_feature_means.csv
    rq4_model_results.csv
    fig1_streams_vs_views.html / .png
    fig2_feature_correlations.html / .png
    fig3_group_comparison.png
    fig4_rq4_importances.png
    dashboard.html              interactive combined view (legend click filters album type)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms"]
LOG = []


def say(msg=""):
    print(msg)
    LOG.append(str(msg))


# ----------------------------------------------------------------------------
# 1. Loading and cleaning
# ----------------------------------------------------------------------------
def load_and_clean(path):
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        say(f"WARNING: standard CSV read failed ({e}).")
        say("Retrying with the forgiving parser (malformed rows are skipped).")
        df = pd.read_csv(path, engine="python", on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    say(f"Rows loaded: {len(df)}, columns: {df.shape[1]}")
    if len(df) < 19000:
        say(f"WARNING: expected about 20,700 rows but found {len(df)}. The file may be "
            "truncated or partly unreadable. Re-download it and check its size "
            "(roughly 30 MB for the Kaggle CSV) before trusting the results.")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    say("\nMissing values in key columns (before cleaning):")
    for c in ["Stream", "Views", "Likes", "Comments"] + FEATURES:
        if c in df.columns:
            say(f"  {c}: {int(df[c].isna().sum())}")

    n0 = len(df)
    key = "Uri" if "Uri" in df.columns else None
    df = df.drop_duplicates(subset=key) if key else df.drop_duplicates(subset=["Artist", "Track"])
    say(f"\nDuplicate rows removed: {n0 - len(df)}")

    n1 = len(df)
    df = df.dropna(subset=["Stream", "Views"])
    say(f"Rows dropped for missing Stream/Views: {n1 - len(df)}")

    n2 = len(df)
    df = df[(df["Stream"] > 0) & (df["Views"] > 0)]
    say(f"Rows dropped for zero Stream/Views: {n2 - len(df)}")

    df = df.dropna(subset=[f for f in FEATURES if f in df.columns])
    # Likes/Comments are left as NaN on purpose (not used in the core analysis).
    if "official_video" in df.columns:
        df["official_video"] = df["official_video"].astype("boolean")
    say(f"Final analytic sample: {len(df)} tracks")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["log_streams"] = np.log10(df["Stream"])
    df["log_views"] = np.log10(df["Views"])
    df["spotify_youtube_ratio"] = df["log_streams"] - df["log_views"]

    q_s, q_v = df["Stream"].quantile(0.75), df["Views"].quantile(0.75)
    df["hit_spotify"] = (df["Stream"] >= q_s).astype(int)
    df["hit_youtube"] = (df["Views"] >= q_v).astype(int)
    df["hit_both"] = ((df["hit_spotify"] == 1) & (df["hit_youtube"] == 1)).astype(int)
    say(f"\nTop-quartile cutoffs: streams >= {q_s:,.0f}, views >= {q_v:,.0f}")

    # Over-performance relative to the log-log trend (RQ3)
    slope, intercept = np.polyfit(df["log_streams"], df["log_views"], 1)
    df["resid"] = df["log_views"] - (slope * df["log_streams"] + intercept)
    lo, hi = df["resid"].quantile(0.10), df["resid"].quantile(0.90)
    df["group"] = "On trend"
    df.loc[df["resid"] >= hi, "group"] = "YouTube-skewed"
    df.loc[df["resid"] <= lo, "group"] = "Spotify-skewed"
    df.loc[df["hit_both"] == 1, "group"] = "Hit on both"
    say(f"Log-log fit: log_views = {slope:.3f} * log_streams + {intercept:.3f}")
    return df


# ----------------------------------------------------------------------------
# 3. Correlation analysis
# ----------------------------------------------------------------------------
def corr_pair(x, y):
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    return r, p, rho, p_s


def rq1_correlations(df):
    rows = []
    r, p, rho, ps = corr_pair(df["log_streams"], df["log_views"])
    rows.append(("All tracks", "all", len(df), r, p, rho, ps))
    for col in ["Album_type", "official_video", "Licensed"]:
        if col not in df.columns:
            continue
        for val, g in df.groupby(col):
            if len(g) >= 30:
                r, p, rho, ps = corr_pair(g["log_streams"], g["log_views"])
                rows.append((col, str(val), len(g), r, p, rho, ps))
    out = pd.DataFrame(rows, columns=["grouping", "value", "n", "pearson_r", "pearson_p",
                                      "spearman_rho", "spearman_p"])
    say("\nRQ1: streams vs views correlation (log scale)")
    say(out.round(4).to_string(index=False))
    return out


def rq2_feature_correlations(df):
    rows = []
    for f in FEATURES:
        rho_s, p_s = stats.spearmanr(df[f], df["log_streams"])
        rho_v, p_v = stats.spearmanr(df[f], df["log_views"])
        rows.append((f, rho_s, p_s, rho_v, p_v))
    out = pd.DataFrame(rows, columns=["feature", "rho_streams", "p_streams",
                                      "rho_views", "p_views"])
    out["abs_max"] = out[["rho_streams", "rho_views"]].abs().max(axis=1)
    out = out.sort_values("abs_max", ascending=False).drop(columns="abs_max")
    say("\nRQ2: Spearman correlation of audio features with popularity")
    say(out.round(4).to_string(index=False))
    return out


def rq3_group_comparison(df):
    cols = FEATURES + ["log_streams", "log_views"]
    means = df.groupby("group")[cols].mean().round(4)
    counts = df["group"].value_counts()
    say("\nRQ3: group sizes")
    say(counts.to_string())
    say("\nRQ3: mean audio features by group")
    say(means.to_string())
    say("\nRQ3: Kruskal-Wallis across groups (feature differs by group?)")
    for f in FEATURES:
        samples = [g[f].values for _, g in df.groupby("group") if len(g) > 5]
        h, p = stats.kruskal(*samples)
        say(f"  {f}: H={h:.2f}, p={p:.4g}")
    if "Album_type" in df.columns:
        say("\nRQ3: share of each album type within each group (%)")
        say((pd.crosstab(df["group"], df["Album_type"], normalize="index") * 100).round(1).to_string())
    return means


def rq4_models(df):
    X = df[FEATURES].values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, importances = [], {}
    for target in ["hit_spotify", "hit_youtube"]:
        y = df[target].values
        logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                    random_state=42, n_jobs=-1)
        for name, model in [("Logistic regression", logit), ("Random forest", rf)]:
            auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            rows.append((target, name, auc.mean(), auc.std(), acc.mean(), y.mean()))
        rf.fit(X, y)
        importances[target] = pd.Series(rf.feature_importances_, index=FEATURES)
    out = pd.DataFrame(rows, columns=["target", "model", "cv_auc_mean", "cv_auc_std",
                                      "cv_accuracy_mean", "base_rate"])
    say("\nRQ4: can audio features alone predict top-quartile status? (5-fold CV)")
    say(out.round(4).to_string(index=False))
    say("(AUC 0.5 = chance. Accuracy should be compared against the majority-class "
        "rate, 1 - base_rate.)")
    return out, importances


# ----------------------------------------------------------------------------
# 4. Visualizations
# ----------------------------------------------------------------------------
def fig1_scatter(df, out):
    fig = px.scatter(df, x="log_streams", y="log_views", color="Danceability",
                     hover_data=["Artist", "Track", "Album_type"], opacity=0.5,
                     color_continuous_scale="Viridis",
                     labels={"log_streams": "Spotify streams (log10)",
                             "log_views": "YouTube views (log10)"},
                     title="Spotify streams vs. YouTube views")
    fig.write_html(os.path.join(out, "fig1_streams_vs_views.html"))
    plt.figure(figsize=(7, 5.5))
    sc = plt.scatter(df["log_streams"], df["log_views"], c=df["Danceability"],
                     s=6, alpha=0.5, cmap="viridis")
    plt.colorbar(sc, label="Danceability")
    slope, intercept = np.polyfit(df["log_streams"], df["log_views"], 1)
    xs = np.linspace(df["log_streams"].min(), df["log_streams"].max(), 100)
    plt.plot(xs, slope * xs + intercept, "r--", lw=1.5, label="Linear fit")
    plt.xlabel("Spotify streams (log10)")
    plt.ylabel("YouTube views (log10)")
    plt.title("Spotify streams vs. YouTube views")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig1_streams_vs_views.png"), dpi=200)
    plt.close()


def fig2_corr_bars(c2, out):
    long = c2.melt(id_vars="feature", value_vars=["rho_streams", "rho_views"],
                   var_name="platform", value_name="spearman_rho")
    long["platform"] = long["platform"].map({"rho_streams": "Spotify streams",
                                             "rho_views": "YouTube views"})
    fig = px.bar(long, x="feature", y="spearman_rho", color="platform", barmode="group",
                 title="Audio feature correlation with popularity (Spearman)")
    fig.write_html(os.path.join(out, "fig2_feature_correlations.html"))
    order = c2["feature"].tolist()
    x = np.arange(len(order))
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - 0.2, c2["rho_streams"], 0.4, label="Spotify streams")
    plt.bar(x + 0.2, c2["rho_views"], 0.4, label="YouTube views")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(x, order, rotation=45, ha="right")
    plt.ylabel("Spearman rho")
    plt.title("Audio feature correlation with popularity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig2_feature_correlations.png"), dpi=200)
    plt.close()


def fig3_group_boxes(df, out):
    feats = ["Danceability", "Energy", "Valence", "Acousticness", "Speechiness", "Loudness"]
    order = ["Spotify-skewed", "On trend", "YouTube-skewed", "Hit on both"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, f in zip(axes.ravel(), feats):
        data = [df.loc[df["group"] == g, f] for g in order]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels([o.replace(" ", "\n") for o in order])
        ax.set_title(f)
    fig.suptitle("Audio features by cross-platform popularity group")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_group_comparison.png"), dpi=200)
    plt.close(fig)


def fig4_importances(importances, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (target, imp) in zip(axes, importances.items()):
        imp.sort_values().plot.barh(ax=ax)
        ax.set_title(f"Random forest importance: {target}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_rq4_importances.png"), dpi=200)
    plt.close(fig)


def dashboard(df, c2, out):
    f1 = px.scatter(df, x="log_streams", y="log_views", color="Album_type",
                    hover_data=["Artist", "Track"], opacity=0.5,
                    labels={"log_streams": "Spotify streams (log10)",
                            "log_views": "YouTube views (log10)"},
                    title="Streams vs. views (click legend to filter album type)")
    long = c2.melt(id_vars="feature", value_vars=["rho_streams", "rho_views"],
                   var_name="platform", value_name="spearman_rho")
    f2 = px.bar(long, x="feature", y="spearman_rho", color="platform", barmode="group",
                title="Feature correlation with popularity")
    f3 = px.box(df, x="Album_type", y="log_views", color="Album_type",
                title="YouTube views by album type")
    html = ("<html><head><title>Streams vs. Screens dashboard</title></head><body>"
            + f1.to_html(full_html=False, include_plotlyjs="cdn")
            + f2.to_html(full_html=False, include_plotlyjs=False)
            + f3.to_html(full_html=False, include_plotlyjs=False)
            + "</body></html>")
    with open(os.path.join(out, "dashboard.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(BASE_DIR, "data", "Spotify_Youtube.csv"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "results"))
    # parse_known_args ignores extra args injected by Colab/Jupyter (e.g. "-f kernel.json")
    args, _ = ap.parse_known_args()
    os.makedirs(args.out, exist_ok=True)

    df = add_features(load_and_clean(args.csv))
    c1 = rq1_correlations(df)
    c2 = rq2_feature_correlations(df)
    means = rq3_group_comparison(df)
    c4, imps = rq4_models(df)

    c1.to_csv(os.path.join(args.out, "rq1_correlations.csv"), index=False)
    c2.to_csv(os.path.join(args.out, "rq2_feature_correlations.csv"), index=False)
    means.to_csv(os.path.join(args.out, "rq3_group_feature_means.csv"))
    c4.to_csv(os.path.join(args.out, "rq4_model_results.csv"), index=False)

    fig1_scatter(df, args.out)
    fig2_corr_bars(c2, args.out)
    fig3_group_boxes(df, args.out)
    fig4_importances(imps, args.out)
    dashboard(df, c2, args.out)

    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    say(f"\nDone. Files written to {args.out}/")


if __name__ == "__main__":
    main()
