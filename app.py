import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATHS = {
    "lr":          "model_lr.pkl",
    "rf":          "model_rf.pkl",
    "lr_fe":       "model_lr_fe.pkl",
    "rf_fe":       "model_rf_fe.pkl",
    "lr_tuned":    "model_lr_tuned.pkl",
    "rf_tuned":    "model_rf_tuned.pkl",
    "lr_fe_tuned": "model_lr_fe_tuned.pkl",
    "rf_fe_tuned": "model_rf_fe_tuned.pkl",
}
METRICS_PATHS = {k: v.replace("model_", "metrics_").replace(".pkl", ".json")
                 for k, v in MODEL_PATHS.items()}

_FALLBACK_DATA = {
    "Survived": [0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,
                 0,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,1,
                 0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,0,1,0,
                 0,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,1],
    "Pclass":   [3,1,3,1,3,3,1,3,3,2,3,1,3,3,3,2,3,2,3,3,
                 2,2,3,1,3,3,1,3,3,3,1,1,3,2,3,1,3,3,3,3,
                 3,3,3,2,3,3,3,1,3,3,3,3,1,2,3,1,3,2,3,3,
                 3,1,3,3,3,2,3,3,1,3,3,3,3,3,2,3,3,3,3,2],
    "Sex":      ["male","female","female","female","male","male","male","male","female","female",
                 "female","female","male","male","female","female","male","male","female","female",
                 "male","female","female","male","female","female","male","male","female","male",
                 "male","female","female","male","male","male","male","male","female","female",
                 "female","female","male","female","female","male","male","female","male","female",
                 "male","male","female","female","male","male","female","male","female","male",
                 "male","female","male","male","male","female","male","male","female","male",
                 "male","male","male","male","female","male","male","male","female","female"],
    "Age":      [22,38,26,35,35,28,54,2,27,14,4,58,20,39,14,55,2,31,31,35,
                 34,15,28,8,38,28,19,28,28,28,40,28,28,28,28,28,28,21,18,14,
                 40,27,28,3,19,28,28,22,28,28,28,28,28,28,28,28,28,28,28,28,
                 28,36,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28],
    "SibSp":    [1,1,0,1,0,0,0,3,0,1,1,0,0,1,0,0,4,0,1,0,
                 0,0,0,3,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,
                 0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "Parch":    [0,0,0,0,0,0,0,1,2,0,1,0,0,5,0,0,1,0,0,0,
                 0,0,0,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "Fare":     [7.25,71.28,7.92,53.10,8.05,8.46,51.86,21.07,11.13,30.07,
                 16.70,26.55,8.05,31.27,7.85,16.00,29.12,13.00,18.00,7.92,
                 26.00,13.00,8.03,35.50,31.39,7.92,10.50,7.25,22.00,9.50,
                 30.00,41.58,15.50,10.50,7.08,9.00,52.00,8.05,18.00,11.24,
                 9.475,21.00,7.04,41.58,7.04,8.05,15.50,15.50,7.58,7.62,
                 7.54,8.05,18.75,19.50,8.05,26.55,8.05,10.50,11.13,0.00,
                 7.92,7.80,7.04,10.50,9.21,9.00,7.92,20.25,26.00,7.25,
                 7.50,8.05,7.85,8.05,9.84,10.50,7.04,12.29,8.05,9.84],
    "Embarked": ["S","C","S","S","S","Q","S","S","S","C","S","S","S","S","S",
                 "S","Q","S","S","C","S","S","Q","S","S","S","C","S","S","S",
                 "S","C","Q","S","S","S","S","S","S","C","S","S","S","S","S",
                 "S","Q","S","S","S","S","S","S","S","S","S","S","S","C","S",
                 "S","C","S","S","Q","S","S","S","C","S","Q","S","S","S","S",
                 "S","S","C","S","S"],
}


def _fetch_df():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame(_FALLBACK_DATA)


def _load_data():
    df = _fetch_df()
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[features + ["Survived"]].dropna()
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)
    X, y = df[features], df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, features


def _load_data_fe():
    df = _fetch_df()
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r',\s*([^.]+)\.').iloc[:, 0].str.strip()
        df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        df["Title"] = df["Title"].apply(
            lambda t: t if t in {"Mr", "Mrs", "Miss", "Master"} else "Rare"
        )
    else:
        df["Title"] = df.apply(
            lambda r: ("Master" if r["Age"] <= 14 else "Mr") if r["Sex"] == "male"
                      else ("Miss" if r["Age"] <= 14 else "Mrs"),
            axis=1,
        )
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["AgeBin"]     = pd.cut(df["Age"], bins=[-1, 12, 17, 60, 100], labels=False)
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)
    df["Title"]    = df["Title"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}).fillna(4)
    features = ["Pclass", "Sex", "AgeBin", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize"]
    df = df[features + ["Survived"]].dropna()
    X, y = df[features], df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, features


def _build_pipeline(model_type):
    if "rf" in model_type:
        return Pipeline([("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42)),
    ])


def _build_param_grid(model_type):
    if "rf" in model_type:
        return {
            "model__n_estimators":      [50, 100, 200],
            "model__max_depth":         [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf":  [1, 2],
        }
    return {
        "model__C":        [0.01, 0.1, 1, 10, 100],
        "model__solver":   ["lbfgs", "liblinear"],
        "model__max_iter": [500, 1000],
    }


def _compute_metrics(pipeline, X_test, y_test, features, model_type):
    from datetime import datetime
    trained_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    accuracy   = pipeline.score(X_test, y_test)
    y_pred     = pipeline.predict(X_test)
    y_proba    = pipeline.predict_proba(X_test)[:, 1]
    cm         = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc    = auc(fpr, tpr)

    is_rf    = "rf"    in model_type
    is_fe    = "fe"    in model_type
    is_tuned = "tuned" in model_type

    base_name    = "Random Forest"          if is_rf    else "Logistic Regression"
    fe_suffix    = " (FE)"                  if is_fe    else ""
    tuned_suffix = " + Tuned"               if is_tuned else ""
    model_name   = base_name + fe_suffix + tuned_suffix

    if is_rf:
        feature_values      = pipeline.named_steps["model"].feature_importances_.tolist()
        feature_chart_label = "Importance"
    else:
        feature_values      = pipeline.named_steps["model"].coef_[0].tolist()
        feature_chart_label = "Coefficient value"

    return {
        "model_name":          model_name,
        "trained_at":          trained_at,
        "feature_chart_label": feature_chart_label,
        "accuracy":            round(float(accuracy), 4),
        "precision":           round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":              round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":                  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix":    cm.tolist(),
        "fpr":                 [round(x, 4) for x in fpr.tolist()],
        "tpr":                 [round(x, 4) for x in tpr.tolist()],
        "roc_auc":             round(float(roc_auc), 4),
        "feature_names":       list(features),
        "coefficients":        [round(x, 4) for x in feature_values],
    }


def train_model(model_type, X_train, X_test, y_train, y_test, features):
    pipeline = _build_pipeline(model_type)

    if "tuned" in model_type:
        search = GridSearchCV(
            pipeline, _build_param_grid(model_type),
            cv=5, scoring="accuracy", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best       = search.best_estimator_
        best_params = search.best_params_
        print(f"[{model_type.upper()}] Best params: {best_params}")
    else:
        pipeline.fit(X_train, y_train)
        best       = pipeline
        best_params = {}

    print(f"[{model_type.upper()}] Test accuracy: {best.score(X_test, y_test):.2%}")

    with open(MODEL_PATHS[model_type], "wb") as f:
        pickle.dump(best, f)

    metrics = _compute_metrics(best, X_test, y_test, features, model_type)
    metrics["best_params"] = best_params
    with open(METRICS_PATHS[model_type], "w") as f:
        json.dump(metrics, f)

    return best, metrics


# ── Startup: load or train all eight models ───────────────────────────────────
BASE_TYPES = ("lr", "rf", "lr_tuned", "rf_tuned")
FE_TYPES   = ("lr_fe", "rf_fe", "lr_fe_tuned", "rf_fe_tuned")

need_base = any(
    not (os.path.exists(MODEL_PATHS[t]) and os.path.exists(METRICS_PATHS[t]))
    for t in BASE_TYPES
)
need_fe = any(
    not (os.path.exists(MODEL_PATHS[t]) and os.path.exists(METRICS_PATHS[t]))
    for t in FE_TYPES
)

if need_base:
    _X_tr, _X_te, _y_tr, _y_te, _feats = _load_data()
if need_fe:
    _X_tr_fe, _X_te_fe, _y_tr_fe, _y_te_fe, _feats_fe = _load_data_fe()

models      = {}
all_metrics = {}

for mtype in BASE_TYPES:
    if os.path.exists(MODEL_PATHS[mtype]) and os.path.exists(METRICS_PATHS[mtype]):
        with open(MODEL_PATHS[mtype], "rb") as f:
            models[mtype] = pickle.load(f)
        with open(METRICS_PATHS[mtype], "r") as f:
            all_metrics[mtype] = json.load(f)
    else:
        models[mtype], all_metrics[mtype] = train_model(
            mtype, _X_tr, _X_te, _y_tr, _y_te, _feats
        )

for mtype in FE_TYPES:
    if os.path.exists(MODEL_PATHS[mtype]) and os.path.exists(METRICS_PATHS[mtype]):
        with open(MODEL_PATHS[mtype], "rb") as f:
            models[mtype] = pickle.load(f)
        with open(METRICS_PATHS[mtype], "r") as f:
            all_metrics[mtype] = json.load(f)
    else:
        models[mtype], all_metrics[mtype] = train_model(
            mtype, _X_tr_fe, _X_te_fe, _y_tr_fe, _y_te_fe, _feats_fe
        )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stats")
def stats():
    return render_template("stats.html")

@app.route("/fe-stats")
def fe_stats():
    return render_template("fe_stats.html")

@app.route("/compare")
def compare():
    return render_template("compare_stats.html")

@app.route("/tuning-stats")
def tuning_stats():
    return render_template("tuning_stats.html")

@app.route("/metrics")
def metrics():
    mtype = request.args.get("model", "lr")
    if mtype not in all_metrics:
        return jsonify({"error": "Unknown model"}), 400
    return jsonify(all_metrics[mtype])

@app.route("/previous-models")
def previous_models():
    return render_template("history.html")

@app.route("/history-data")
def history_data():
    history_path = "model_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route("/save-model", methods=["POST"])
def save_model():
    from datetime import date
    data  = request.get_json()
    name  = (data.get("name") or "Model").strip()
    mtype = data.get("model", "lr")
    if mtype not in all_metrics:
        return jsonify({"error": "Unknown model"}), 400

    m = all_metrics[mtype]
    entry = {
        "name":             name,
        "date":             date.today().strftime("%d/%m/%Y"),
        "accuracy":         m["accuracy"],
        "precision":        m["precision"],
        "recall":           m["recall"],
        "f1":               m["f1"],
        "auc":              m["roc_auc"],
        "confusion_matrix": m["confusion_matrix"],
        "notes":            "",
    }

    history_path = "model_history.json"
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    history.append(entry)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    data  = request.get_json()
    mtype = data.get("model", "lr")
    if mtype not in models:
        return jsonify({"error": "Unknown model"}), 400

    try:
        pclass   = int(data["pclass"])
        sex      = 0 if data["sex"] == "male" else 1
        age      = float(data["age"])
        sibsp    = int(data["sibsp"])
        parch    = int(data["parch"])
        fare     = float(data["fare"])
        embarked = {"S": 0, "C": 1, "Q": 2}.get(data["embarked"], 0)

        if "fe" in mtype:
            age_bin     = 0 if age <= 12 else 1 if age <= 17 else 2 if age <= 60 else 3
            family_size = sibsp + parch + 1
            title = (3 if age <= 14 else 0) if sex == 0 else (2 if age <= 14 else 1)
            feat_arr = np.array([[pclass, sex, age_bin, sibsp, parch, fare, embarked, title, family_size]])
        else:
            feat_arr = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

        prediction  = models[mtype].predict(feat_arr)[0]
        probability = models[mtype].predict_proba(feat_arr)[0]

        return jsonify({
            "survived":             bool(prediction),
            "survived_probability": round(float(probability[1]) * 100, 1),
            "died_probability":     round(float(probability[0]) * 100, 1),
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
