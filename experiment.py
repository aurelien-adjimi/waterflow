import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import xgboost as xgb
import seaborn as sns
from itertools import product

df_clean = pd.read_csv("data/df_clean.csv")

X = df_clean.drop("Potability", axis=1).values
y = df_clean["Potability"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

param_grid_keras = {
    "dense1": [128, 64],
    "dropout1": [0.4, 0.3],
    "dense2": [64],
    "dropout2": [0.3, 0.5],
    "epochs": [50, 100]
}

mlflow.set_experiment("experiment_water_quality_mlp")



for dense1, dropout1, dense2, dropout2, epochs in product(
    param_grid_keras["dense1"],
    param_grid_keras["dropout1"],
    param_grid_keras["dense2"],
    param_grid_keras["dropout2"],
    param_grid_keras["epochs"]):

    with mlflow.start_run(run_name=f"mlp_{dense1}_{dropout1}"):
        model = Sequential()
        model.add(Dense(dense1, activation='relu', input_shape=(X.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout1))
        model.add(Dense(dense2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_param("dense1", dense1)
        mlflow.log_param("dropout1", dropout1)
        mlflow.log_param("dense2", dense2)
        mlflow.log_param("dropout2", dropout2)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", acc)
        mlflow.keras.log_model(model, artifact_path="model")


param_grid_xgb = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5]
} 


mlflow.set_experiment("experiment_water_quality_xg")


X = df_clean.drop("Potability", axis=1).values
y = df_clean["Potability"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


for n_estimators, learning_rate, max_depth in product(
        param_grid_xgb["n_estimators"],
        param_grid_xgb["learning_rate"],
        param_grid_xgb["max_depth"]):
    
    with mlflow.start_run(run_name=f"xgb_{n_estimators}_{learning_rate}_{max_depth}"):
        model_xgb = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        model_xgb.fit(X_train, y_train)
        y_pred = model_xgb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model_xgb, "model")

