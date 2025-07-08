import mlflow.keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def test_keras_model_can_load():
    model = mlflow.keras.load_model("models:/mlp_keras_best_model/Production")
    assert model is not None


def test_scaler_output_range():
    df = pd.read_csv("data/df_clean.csv")
    X = df.drop("Potability", axis=1).values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    assert np.all(X_scaled >= -1e-8) and np.all(X_scaled <= 1 + 1e-8), \
        "Le scaler ne ramène pas dans la plage [0, 1] (tolérance flottante incluse)"

