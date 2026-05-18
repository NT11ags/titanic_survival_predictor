# 🚢 Titanic Survival Predictor

A Flask web app that predicts Titanic passenger survival using logistic regression trained on the classic Titanic dataset.

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

## How it works

1. **Model**: scikit-learn `LogisticRegression` with `StandardScaler` in a `Pipeline`
2. **Dataset**: Titanic CSV fetched from GitHub on first build; fallback embedded dataset included
3. **Features used**: Passenger class, Sex, Age, Siblings/Spouse, Parents/Children, Fare, Port of embarkation
4. **Server**: Gunicorn with 2 workers — production-ready

## Project Structure
```
titanic-app/
├── app.py               # Flask app + model training
├── templates/
│   └── index.html       # Styled frontend
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
