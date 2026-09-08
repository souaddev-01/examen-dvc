import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_DIR = "data/processed"

X_train = pd.read_csv(f"{INPUT_DIR}/X_train.csv")
X_test = pd.read_csv(f"{INPUT_DIR}/X_test.csv")

# Conversion de la colonne date en variables numériques
for df in [X_train, X_test]:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["hour"] = df["date"].dt.hour
        df.drop(columns=["date"], inplace=True)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

X_train_scaled.to_csv(
    f"{INPUT_DIR}/X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    f"{INPUT_DIR}/X_test_scaled.csv",
    index=False
)

print("Normalisation terminée.")
print("X_train_scaled :", X_train_scaled.shape)
print("X_test_scaled  :", X_test_scaled.shape)
