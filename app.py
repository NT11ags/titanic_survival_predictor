import io
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = "model.pkl"


def train_model():
    """Train logistic regression on the Titanic dataset."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback: embedded minimal dataset for offline use
        data = {
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
        df = pd.DataFrame(data)

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[features + ["Survived"]].dropna()

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)

    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {accuracy:.2%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    return pipeline


# Train or load model at startup
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        pclass = int(data["pclass"])
        sex = 0 if data["sex"] == "male" else 1
        age = float(data["age"])
        sibsp = int(data["sibsp"])
        parch = int(data["parch"])
        fare = float(data["fare"])
        embarked = {"S": 0, "C": 1, "Q": 2}.get(data["embarked"], 0)

        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        survived_prob = float(probability[1])
        died_prob = float(probability[0])

        return jsonify({
            "survived": bool(prediction),
            "survived_probability": round(survived_prob * 100, 1),
            "died_probability": round(died_prob * 100, 1),
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
