from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    def conv(valor):
        return 1 if valor.lower() == "sim" else 0

    data = [
        conv(request.form["agua"]),
        conv(request.form["creme"]),
        conv(request.form["escovacao"]),
        conv(request.form["fio"]),
        conv(request.form["noite"]),
        conv(request.form["acucar"]),
        conv(request.form["carbo"]),
        conv(request.form["odonto"]),
        conv(request.form["aparelho"]),
        conv(request.form["enxaguante"])
    ]

    X = np.array(data).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    return render_template(
        "index.html",
        prediction=pred,
        probs=probs
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)