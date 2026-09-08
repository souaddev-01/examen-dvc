import pandas as pd
import pickle
import json
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INPUT_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_DIR = "metrics"

os.makedirs(METRICS_DIR, exist_ok=True)

X_test = pd.read_csv(f"{INPUT_DIR}/X_test_scaled.csv")
y_test = pd.read_csv(f"{INPUT_DIR}/y_test.csv").squeeze()

with open(f"{MODEL_DIR}/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

predictions = pd.DataFrame({
    "actual": y_test,
    "prediction": y_pred
})

predictions.to_csv("data/predictions.csv", index=False)

scores = {
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2
}

with open(f"{METRICS_DIR}/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("Évaluation terminée.")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R2   : {r2:.4f}")
print("Prédictions sauvegardées dans data/predictions.csv")
print("Métriques sauvegardées dans metrics/scores.json")
