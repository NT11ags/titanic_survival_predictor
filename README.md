# 🚢 Titanic Survival Predictor

A Flask web app that predicts Titanic passenger survival using multiple machine learning models trained on the classic Titanic dataset.

## Quick Start

### Option A — Docker Compose (recommended)
```bash
docker compose up --build
```

### Option B — Docker CLI
```bash
docker build -t titanic-predictor .
docker run -p 5000:5000 titanic-predictor
```

Then open **http://localhost:5000** in your browser.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| ML | scikit-learn (LogisticRegression, RandomForestClassifier, VotingClassifier, GridSearchCV, Pipeline) |
| Data | pandas, NumPy |
| Frontend | HTML/CSS/JS, Chart.js v4 |
| Fonts | Google Fonts (Cormorant Garamond, Courier Prime) |
| Server | Gunicorn |
| Containerisation | Docker, Docker Compose |

## Features

- **Survival predictor** — enter passenger details and get an instant survival probability from any of the 10 trained models
- **10 model variants** — Logistic Regression and Random Forest, each available in base, feature-engineered, hyperparameter-tuned, and ensemble (VotingClassifier) forms
- **Feature engineering** — title extraction from passenger name, family size, and age binning added as additional features
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross-validation automatically finds optimal parameters for each model
- **Ensemble models** — soft-voting VotingClassifiers combine LR and RF predictions for base, FE, and tuned variants
- **Interactive stats pages** — metrics bar charts, ROC curves, confusion matrices, and feature importance/coefficient charts for every model
- **Model comparison pages** — side-by-side views for base vs FE, untuned vs tuned, and sub-models vs ensemble
- **Model history** — save any trained model's stats to a persistent archive with a custom name
- **Persistent models** — trained models are cached to disk and reloaded on restart, skipping retraining

## Results

### Ensemble models outperformed individual models
The soft-voting `VotingClassifier` ensembles (combining Logistic Regression and Random Forest) consistently achieved higher accuracy than either model alone. Averaging the predicted probabilities from two different model types reduced individual errors, producing more reliable predictions across the board.

### Feature engineering made little difference
Extracting passenger titles, creating a family size feature, and binning age into groups did not meaningfully improve performance over the base 7-feature models. The original features (class, sex, age, fare, etc.) already captured most of the signal in the dataset.

### Hyperparameter tuning had minimal impact
Running `GridSearchCV` over Logistic Regression and Random Forest hyperparameters produced negligible gains compared to the default settings. The Titanic dataset is small and relatively simple, so the default hyperparameters were already close to optimal and additional tuning had little room to improve on them.

## How it works

1. **Models**: Logistic Regression and Random Forest, in base, feature-engineered, tuned, and ensemble variants (10 models total)
2. **Dataset**: Titanic CSV fetched from GitHub on first build; fallback embedded dataset included
3. **Features used**: Passenger class, Sex, Age, Siblings/Spouse, Parents/Children, Fare, Port of embarkation
4. **Hyperparameter tuning**: `GridSearchCV` with 5-fold cross-validation
5. **Ensemble**: Soft-voting `VotingClassifier` combining LR and RF predictions
6. **Server**: Gunicorn with 2 workers — production-ready

## Project Structure
```
titanic-app/
├── app.py                        # Flask app + model training
├── templates/
│   ├── index.html                # Predictor UI
│   ├── stats.html                # Base model comparison
│   ├── fe_stats.html             # Feature-engineered models
│   ├── compare_stats.html        # Base vs FE comparison
│   ├── tuned_stats.html          # Tuned model comparison
│   ├── tuning_stats.html         # Untuned vs tuned comparison
│   ├── ensemble_stats.html       # Ensemble model stats
│   └── history.html              # Saved model archive
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
