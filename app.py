from flask import Flask, request, jsonify, render_template
import mlflow.keras
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_URI = "models:/mlp_keras_best_model/Production"
model = mlflow.keras.load_model(MODEL_URI)
scaler = joblib.load("data/scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            features = [
                float(request.form.get("ph")),
                float(request.form.get("Hardness")),
                float(request.form.get("Solids")),
                float(request.form.get("Chloramines")),
                float(request.form.get("Sulfate")),
                float(request.form.get("Conductivity")),
                float(request.form.get("Organic_carbon")),
                float(request.form.get("Trihalomethanes")),
                float(request.form.get("Turbidity"))
            ]
            df = pd.DataFrame([features], columns=[
                "ph", "Hardness", "Solids", "Chloramines",
                "Sulfate", "Conductivity", "Organic_carbon",
                "Trihalomethanes", "Turbidity"
            ])
            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0][0]
            result = "Potable" if prediction >= 0.5 else "Non potable"
            return render_template("index.html", prediction=result)
        except Exception as e:
            return render_template("index.html", prediction="Erreur : " + str(e))
    return render_template("index.html")


if __name__ == "__main__":
    print("Application Flask en cours d'exécution sur http://127.0.0.1:5001")
    app.run(debug=True, port=5001)