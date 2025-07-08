import mlflow.keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def test_keras_model_predicts_output():
    model = mlflow.keras.load_model("models:/mlp_keras_best_model/Production")

    df = pd.read_csv("data/df_clean.csv")
    X = df.drop("Potability", axis=1).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    preds = model.predict(X_scaled)

    assert preds.shape[0] == X_scaled.shape[0], "Nombre de prédictions incorrect"
    assert ((preds >= 0) & (preds <= 1)).all(), "Les prédictions doivent être entre 0 et 1"

def test_prediction_output_type():
    import numpy as np
    model = mlflow.keras.load_model("models:/mlp_keras_best_model/Production")
    sample = np.array([[0.3] * 9])
    pred = model.predict(sample)
    
    assert pred.shape == (1, 1), "La prédiction n'est pas du bon format"
    assert 0 <= pred[0][0] <= 1, "La prédiction doit être une probabilité entre 0 et 1"
