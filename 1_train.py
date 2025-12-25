import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "food_clean_example.csv"

MODEL_DIR = BASE_DIR / "🤖model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# целевая переменная: хорошее блюдо
df["is_good"] = (df["rating"] >= 4.0).astype(int)

# числовые базовые признаки
num_cols = ["rating", "calories", "protein", "fat", "sodium"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=["title"])

# инженерные фичи
df["prot_kcal"] = df["protein"] / (df["calories"] + 0.001)
df["fat_kcal"] = df["fat"] / (df["calories"] + 0.001)
df["salt_kcal"] = df["sodium"] / (df["calories"] + 0.001)

df["prot_fat_ratio"] = df["protein"] / (df["fat"] + 0.1)

if "weight" in df.columns:
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["weight"] = df["weight"].fillna(df["weight"].median())
    df["cal_density"] = df["calories"] / (df["weight"] + 1)
else:
    df["cal_density"] = df["calories"]  # заглушка

df["salt_density"] = df["sodium"] / (df["calories"] + 1)

# one-hot по категориальным признакам
cat_cols = []
if "cuisine" in df.columns:
    cat_cols.append("cuisine")
if "category" in df.columns:
    cat_cols.append("category")

if cat_cols:
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

# формируем X, y
target_col = "is_good"
y = df[target_col].values

feature_cols = [c for c in df.columns if c not in ["title", target_col]]

X = df[feature_cols].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(feature_cols, MODEL_DIR / "features.joblib")
print("Модель переобучена и сохранена.")
print(f"Всего блюд: {len(df)}, accuracy: {acc:.3f}")
