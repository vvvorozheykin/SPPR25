import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("🤖model/scaler.joblib")
model = joblib.load("🤖model/model.pkl")
features = joblib.load("🤖model/features.joblib")

def recommend(max_cal=9999, min_prot=0, top=5):
    df = pd.read_csv("food.csv")
    num_cols = ["rating", "calories", "protein", "fat", "sodium"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["title"])

    if max_cal < 9999:
        df = df[df["calories"] <= max_cal]
    if min_prot > 0:
        df = df[df["protein"] >= min_prot]

    df["prot_kcal"] = df["protein"] / (df["calories"] + 0.001)
    df["fat_kcal"] = df["fat"] / (df["calories"] + 0.001)
    df["salt_kcal"] = df["sodium"] / (df["calories"] + 0.001)

    X = df[features].values.astype(np.float32)
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    df["score"] = probs
    return df.sort_values("score", ascending=False).head(top)

if __name__ == "__main__":
    print(recommend())
