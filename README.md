# 🚢 Titanic Survival Predictor

A Flask web app that predicts Titanic passenger survival using multiple machine learning models trained on the classic Titanic dataset.

**Live demo:** https://titanic-survival-predictor-x6xa.onrender.com/

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
| ML | scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC, KNeighborsClassifier, VotingClassifier, StackingClassifier, GridSearchCV, Pipeline) |
| Data | pandas, NumPy |
| Frontend | HTML/CSS/JS, Chart.js v4 |
| Fonts | Google Fonts (Cormorant Garamond, Courier Prime) |
| Server | Gunicorn |
| Containerisation | Docker, Docker Compose |

## Features

- **Survival predictor** — enter passenger details and get an instant survival probability from any of the 11 trained models
- **11 model variants** — Logistic Regression and Random Forest in base, feature-engineered, hyperparameter-tuned, and ensemble forms, plus a stacking ensemble
- **Feature engineering** — title extraction from passenger name, family size, and age binning added as additional features
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross-validation automatically finds optimal parameters for each model
- **Ensemble models** — soft-voting VotingClassifiers combining LR and RF, plus a StackingClassifier with five diverse base estimators and an LR meta-learner
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

### Stacking did not significantly improve performance
A `StackingClassifier` combining five diverse base estimators — Logistic Regression, Random Forest, Gradient Boosting, Support Vector Classifier, and K-Nearest Neighbours — with a Logistic Regression meta-learner produced results comparable to the simpler soft-voting ensembles. The added complexity of stacking (out-of-fold meta-feature generation, a trained meta-learner) yielded no meaningful gain. This reinforces the broader pattern: the Titanic dataset is small enough and its signal straightforward enough that the choice of combination strategy matters less than the underlying feature set.

## How it works

1. **Models**: Logistic Regression and Random Forest in base, feature-engineered, tuned, and ensemble variants, plus a stacking ensemble (11 models total)
2. **Dataset**: Titanic CSV fetched from GitHub on first build; fallback embedded dataset included
3. **Features used**: Passenger class, Sex, Age, Siblings/Spouse, Parents/Children, Fare, Port of embarkation
4. **Hyperparameter tuning**: `GridSearchCV` with 5-fold cross-validation
5. **Voting ensembles**: Soft-voting `VotingClassifier` combining LR and RF predictions
6. **Stacking ensemble**: `StackingClassifier` with LR, RF, GBM, SVC, and KNN base estimators; LR meta-learner trained on out-of-fold predictions
7. **Server**: Gunicorn with 2 workers — production-ready

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
│   ├── ensemble_stats.html       # Voting ensemble stats
│   ├── stack_stats.html          # Stacking ensemble stats
│   └── history.html              # Saved model archive
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
