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

# 1. Charger les données nettoyées
df_clean = pd.read_csv("data/df_clean.csv")

# 2. Préparation des données
X = df_clean.drop("Potability", axis=1).values
y = df_clean["Potability"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Lancement de l'expérience MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("experiment_water_quality_mlp")
with mlflow.start_run(run_name="keras_mlp_cleaned"):

    # Modèle MLP identique à celui du notebook
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )


mlflow.set_experiment("experiment_water_quality_xg")
X = df_clean.drop("Potability", axis=1).values
y = df_clean["Potability"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model_xgb.fit(X_train, y_train)


y_pred = model_xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred)
print(f"\n Accuracy XGBoost : {acc_xgb * 100:.2f}%")

print("\n Rapport de classification :")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Prédiction")
plt.ylabel("Réel")
plt.title("Matrice de confusion - XGBoost")
plt.show()