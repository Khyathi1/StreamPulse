StreamPulse 🎵📊
Cross-Platform Music Popularity Analytics
StreamPulse is an interactive data analysis and visualization project exploring how music popularity on Spotify relates to engagement on YouTube. It combines statistical analysis, audio-feature exploration, cross-platform performance comparisons, and machine-learning experiments in a Streamlit dashboard.

The project focuses on understanding patterns in music engagement—not claiming that audio features alone determine whether a song becomes a hit.
Contents
- Overview
- Research Questions
- Features
- Dataset
- Technology Stack
- Project Structure
- Getting Started
- Running the Project
- Analysis Methods
- Key Findings
- Outputs
- Limitations
- Future Improvements
Overview
Music can perform differently across streaming and video platforms. A track with many Spotify streams may not receive a proportional number of YouTube views, and audio characteristics may have different relationships with engagement on each platform.

StreamPulse investigates these patterns using descriptive statistics, correlation analysis, distribution comparisons, and classification models. The dashboard lets users explore the data interactively, compare groups of tracks, and inspect model results.
Research Questions
RQ1 — Do popular tracks on Spotify also tend to be popular on YouTube?
- Compare Spotify streams with YouTube video views.
- Use log-transformed counts to make highly skewed popularity distributions easier to examine.
- Calculate Pearson and Spearman correlations.
- Compare the relationship across album types and official-video status when those fields are available.
RQ2 — Which audio features are associated with popularity on each platform?
Examine the relationship between ten audio features and Spotify streams or YouTube views:
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
The dashboard presents correlation summaries and feature-specific density heatmaps.
RQ3 — How do tracks that overperform on one platform differ?
Compare four groups derived from performance relative to the Spotify–YouTube trend:
- Spotify-skewed: tracks in the lower tail of the trend residuals.
- On trend: tracks that are not classified into an overperformance tail or the dual-hit group.
- YouTube-skewed: tracks in the upper tail of the trend residuals.
- Hit on both: tracks in the top quartile for both Spotify streams and YouTube views.
Compare audio-feature distributions, group means, and album-type composition.
RQ4 — How well can audio features predict hit status?
Evaluate whether the ten audio features can classify top-quartile popularity on Spotify and YouTube. The implemented models are:
- Logistic regression
- Random forest
Model evaluation uses stratified five-fold cross-validation and reports ROC AUC and accuracy. Random-forest feature importance is also displayed.
Features
- Interactive Streamlit dashboard with Overview, RQ1, RQ2, RQ3, RQ4, and Data tabs.
- Flexible data input through CSV upload or a local file path.
- Data filters for album type and official-video status.
- Cross-platform correlation analysis using Pearson and Spearman statistics.
- Interactive scatterplot of Spotify streams versus YouTube views, including a fitted trend line.
- Audio-feature explorer with correlation charts, tables, and density heatmaps.
- Overperformer analysis with group sizes, boxplots, summary statistics, and album-type shares.
- Hit-prediction experiments using logistic regression and random forest.
- Downloadable filtered data from the dashboard.
- Standalone analysis script that saves tables, figures, and a summary of results.
Dataset
The project is designed for the Spotify and YouTube dataset published on Kaggle:

Spotify and YouTube Dataset — Kaggle

The dataset includes track metadata, Spotify stream counts, YouTube views, and audio characteristics. Exact columns and row counts can vary by dataset version.
Set up the data
1. Download the CSV from the dataset source.
2. Create a data/ directory in the project root.
3. Place the file at data/Spotify_Youtube.csv.
The default dashboard path is data/Spotify_Youtube.csv. Alternatively, upload the CSV from the dashboard sidebar.

Data note: The raw dataset is not required to be committed to GitHub. Check the dataset's license and terms before redistributing it. For a cleaner repository, keep the CSV locally and document how to obtain it.
Technology Stack
Area	Tools
Language	Python
Data manipulation	pandas, NumPy
Statistical analysis	SciPy
Machine learning	scikit-learn
Interactive visualizations	Plotly
Static visualizations	Matplotlib
Dashboard	Streamlit
Version control	Git and GitHub


Project Structure
StreamPulse/
├── app.py              # Interactive Streamlit dashboard
├── analysis.py         # Analysis pipeline and output generation
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Files excluded from version control
├── data/
│   └── Spotify_Youtube.csv   # Local dataset; do not commit by default
└── results/            # Generated tables, figures, and summaries
The results/ directory is created by the analysis script when it runs.
Getting Started
Requirements
- Python 3.10 or newer recommended
- pip
- The project CSV dataset
1. Clone the repository
Replace YOUR_USERNAME with your GitHub username after creating the repository.
git clone https://github.com/YOUR_USERNAME/StreamPulse.git
cd StreamPulse
2. Create and activate a virtual environment
Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Add the dataset
Put Spotify_Youtube.csv inside the data/ directory, or upload it through the dashboard sidebar.
Running the Project
Launch the interactive dashboard
From the project root:
streamlit run app.py
Streamlit will display a local URL in the terminal, usually http://localhost:8501. Open that address in a browser.
Run the analysis pipeline
python analysis.py
To specify the input CSV and output directory:
python analysis.py --csv data/Spotify_Youtube.csv --out results
The analysis script writes a summary, result tables, and visualizations to the output directory.
Analysis Methods
Data preparation
The pipeline removes unnamed index columns, removes duplicate tracks, excludes rows missing stream/view counts, excludes nonpositive stream/view values before log transformations, and drops records missing the audio features required for analysis.
Feature engineering
- Calculates log10(Stream) and log10(Views).
- Defines Spotify and YouTube hit labels using each platform's 75th-percentile threshold.
- Fits a linear trend between log streams and log views.
- Uses trend residuals to identify platform-skewed groups.
Statistical analysis
- Pearson correlation measures linear association.
- Spearman correlation measures rank-based monotonic association.
- Group summaries compare average audio features and album-type composition.
- Kruskal–Wallis tests are included in the analysis pipeline to compare feature distributions across groups.
Predictive modeling
Logistic regression and random forest models use the ten audio features to predict top-quartile hit labels. Stratified five-fold cross-validation evaluates ROC AUC and accuracy. A majority-class accuracy baseline is useful for interpreting the reported accuracy.
Key Findings
In the analyzed dataset version, the current dashboard run showed the following patterns:
- Streams and views are positively associated. The overall Spearman correlation shown in the dashboard was approximately 0.611.
- Official-video status matters to the observed relationship. The screenshot showed a stronger streams–views correlation for tracks marked official than for tracks marked not official.
- Loudness had the strongest positive audio-feature correlation with YouTube views among the ten features shown, while acousticness and instrumentalness had negative correlations.
- The overperformer groups differed in their average audio features and album-type composition.
- Audio-feature-only hit prediction was limited. In the displayed run, the highest AUC was approximately 0.675 for YouTube hit prediction using random forest. Accuracy was close to the majority-class baseline.
These values describe a particular analyzed dataset and dashboard configuration. Re-run the pipeline on your own data file to reproduce and verify the results before using them elsewhere.
Outputs
The standalone analysis script can generate:
- rq1_correlations.csv — overall and subgroup streams-versus-views correlations.
- rq2_feature_correlations.csv — audio-feature correlation results.
- rq3_group_feature_means.csv — group-level feature summaries.
- rq4_model_results.csv — classification metrics.
- fig1_streams_vs_views.png and .html — streams-versus-views plot.
- fig2_feature_correlations.png and .html — audio-feature correlation chart.
- fig3_group_comparison.png — feature distributions across performance groups.
- fig4_rq4_importances.png — random-forest feature importance.
- dashboard.html — combined interactive visualization.
Generated files appear in the output directory specified to analysis.py (by default, results/).
Limitations
- Correlation does not establish causation.
- Audio features alone cannot capture all drivers of music popularity, such as artist reach, promotion, release age, playlists, and video-channel audience.
- Hit labels are relative to the dataset's top quartile, not universal measures of commercial success.
- Trend-based groups are analytical categories rather than official industry classifications.
- Predictive results depend on the dataset, filters, preprocessing, and cross-validation setup.
- Feature importance describes how a model uses variables; it does not establish that those variables cause popularity.
Future Improvements
- Add confidence intervals and more detailed model diagnostics.
- Compare models with and without audio features and test additional baselines.
- Evaluate how release date, artist reach, and video engagement metrics affect performance, if reliable data are available.
- Improve dashboard layout and add export options for charts and filtered results.
- Add automated tests for data cleaning and analysis functions.
- Deploy the dashboard so it can be explored without installing Python locally.
Reproducibility
For reproducible results, use the same dataset version, install the listed dependencies, preserve the analysis configuration, and record any dashboard filters used when interpreting model metrics.
