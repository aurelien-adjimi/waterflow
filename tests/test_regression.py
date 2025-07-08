import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

def test_keras_model_accuracy_threshold():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.keras.load_model("models:/mlp_keras_best_model/Production")

    df = pd.read_csv("data/df_clean.csv")
    X = df.drop("Potability", axis=1).values
    y = df["Potability"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    preds = model.predict(X_scaled)
    y_pred_class = (preds > 0.5).astype(int)

    acc = accuracy_score(y, y_pred_class)
    assert acc > 0.60, f"Accuracy trop basse : {acc:.2f}"


