import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

INPUT_DIR = "data/processed"
OUTPUT_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")
y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").squeeze()

model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Meilleurs paramètres :")
print(grid_search.best_params_)

print("Meilleur score :")
print(grid_search.best_score_)

with open(f"{OUTPUT_DIR}/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print("Paramètres sauvegardés dans models/best_params.pkl")
