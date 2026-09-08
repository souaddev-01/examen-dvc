import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor

INPUT_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")
y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").squeeze()

with open(f"{MODEL_DIR}/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Entraînement terminé.")
print("Modèle sauvegardé dans models/model.pkl")
