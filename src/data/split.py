import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT = "data/raw/raw.csv"
OUTPUT_DIR = "data/processed"
TARGET = "silica_concentrate"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

print("Split terminé.")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
