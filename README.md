# Heart Disease Risk Analysis

End-to-end Streamlit and machine learning project for UCI Heart Disease risk
analysis. The structure follows the same pattern as the customer churn project:
offline training saves a model artifact, and Streamlit loads that artifact for
interactive prediction and visual analysis.

## Overview

The app explores healthcare data, compares supervised classifiers, and provides a
local risk prediction workflow.

```text
raw UCI heart data
  -> preprocessing
  -> binary target creation
  -> train/test split
  -> feature transformation
  -> KNN and Random Forest training
  -> holdout evaluation
  -> artifact persistence
  -> Streamlit prediction and analysis
```

## Problem Definition

The original UCI target column is:

```text
num
```

The dashboard keeps this value for severity analysis. The ML task is supervised
binary classification:

```text
heart_disease = 1 when num > 0
heart_disease = 0 when num = 0
```

The saved model returns a probability. The default threshold is:

```text
probability >= 0.5 -> Heart disease
probability < 0.5  -> No heart disease
```

## Data Contract

Default dataset path:

```text
data/heart_disease_uci.csv
```

Important columns:

- `num`: original UCI disease severity target from 0 to 4
- `id`: row identifier, removed before model training
- `dataset`: source collection site, excluded from training by default
- `age`, `trestbps`, `chol`, `thalch`, `oldpeak`, `ca`: numeric clinical features
- `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`: categorical clinical features

The `dataset` column can improve scores by learning site-specific patterns, but
it may also encode collection bias. The default patient-facing model excludes it.

## Repository Layout

```text
.
|-- .github/
|   `-- workflows/
|       `-- ci.yml                # GitHub Actions training workflow
|-- app/
|   |-- app.py                    # Streamlit UI and app-facing services
|   |-- data_analysis.py          # Dashboard EDA and clinical charts
|   |-- styles.py                 # Streamlit page styling
|   `-- training_methodology.py   # Training notes and saved metrics view
|-- data/
|   |-- heart_disease_uci.csv     # UCI dataset
|   `-- README.md                 # Dataset placement note
|-- models/
|   |-- model.joblib              # Created by training
|   |-- training_metrics.json     # Created by training
|   `-- README.md
|-- notebooks/
|   `-- README.md
|-- src/
|   |-- config.py                 # Runtime paths, labels, threshold
|   |-- model_bundle.py           # Artifact contract and joblib persistence
|   |-- predict.py                # Artifact-backed prediction services
|   |-- preprocessing.py          # Cleaning, labels, feature preparation
|   |-- train.py                  # Offline training workflow
|   `-- training_settings.py      # Model and split settings
|-- Dockerfile.streamlit
|-- docker-compose.yml
|-- requirements.txt              # Runtime dependencies
|-- requirements-train.txt        # Runtime + notebook helpers
`-- streamlit_app.py              # Streamlit entrypoint
```

## Run Locally

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Train the model:

```powershell
python -m src.train
```

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Training

Training compares the candidate models configured in `src/training_settings.py`.
The current candidate list is persisted into `models/training_metrics.json` every
time `python -m src.train` runs.

Both models use the same train/test split and preprocessing contract:

- numeric columns: median imputation followed by `StandardScaler`
- categorical columns: missing category imputation followed by `OneHotEncoder`
- split: stratified holdout split to preserve the disease/no-disease ratio
- train/test rows, random seed, threshold, and model settings are written to
  `models/training_metrics.json`
- the dashboard reads the saved metrics artifact dynamically

### KNN

KNN is a distance-based classifier. Scaling numeric features is important because
features such as cholesterol, age, and ST depression live on different numeric
ranges. The active neighbor count is defined in `TrainingSettings` and copied to
the metrics artifact after training.

Distance weighting lets closer records contribute more strongly than farther
neighbors.

### Random Forest

Random Forest is a tree ensemble. It captures non-linear feature interactions
and supports interpretable feature importance. Tree count, maximum depth, leaf
size, and related settings are read from `TrainingSettings` during training and
stored in `models/training_metrics.json`.

Class balancing helps because the dataset has more disease-positive records than
healthy records after binarizing `num`.

### Dynamic Metrics Artifact

Model performance is not maintained by hand in this README. It is generated by
the training workflow and stored here:

```text
models/training_metrics.json
```

The artifact contains:

- selected model
- metrics for each candidate model
- confusion matrices
- train/test row counts
- feature schema
- prediction threshold
- training settings

F1 is used for selection because precision and recall both matter in a healthcare
risk-screening style task. ROC AUC is still reported because it measures ranking
quality across thresholds. The Streamlit Training Methodology tab reads this file
and renders the current metrics table dynamically.

The selected model defaults to the best holdout F1 score. You can force a model:

```powershell
python -m src.train --model random_forest
python -m src.train --model knn
```

The UCI `dataset` source column is excluded by default to reduce collection-site
bias. Include it for comparison with:

```powershell
python -m src.train --include-dataset-feature
```

You can also change the prediction threshold:

```powershell
python -m src.train --threshold 0.45
```

Lower thresholds usually increase recall and false positives. Higher thresholds
usually increase precision and false negatives.

## GitHub Actions Training Pipeline

The repository includes a GitHub Actions workflow at:

```text
.github/workflows/ci.yml
```

It follows the same idea as the customer churn project:

1. Run on pushes to `main` when training-relevant files change.
2. Set up Python 3.12.
3. Install `requirements-train.txt`.
4. Compile Python files.
5. Run `python -m src.train`.
6. Commit updated `models/model.joblib` and `models/training_metrics.json` if
   retraining changes the artifacts.

The workflow can also be started manually from the GitHub Actions tab with
`workflow_dispatch`.

## Dashboard

The Streamlit app includes:

- live heart disease risk prediction from `models/model.joblib`
- binary target and original severity distribution
- source dataset analysis
- categorical clinical segment analysis
- numeric vitals/labs distributions
- age-band risk trends
- scatter plots and correlation heatmap
- data quality and missing-value profiling
- saved KNN vs Random Forest metrics
- model and metrics download

## Docker

```powershell
docker-compose up --build
```

Then open:

```text
http://localhost:8501
```

## Notes

This project is for ML/statistical analysis and portfolio demonstration only. It
is not a medical diagnosis, treatment, or triage tool.
