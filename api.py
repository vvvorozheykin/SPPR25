from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ==== Пути ====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "🤖model"
DATA_DIR = BASE_DIR / "data"
FOOD_DB = DATA_DIR / "food_clean_example.csv"  # <-- твой файл с блюдами

# ==== FastAPI-приложение ====
app = FastAPI(title="🍽️ Food Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Загрузка модели ====
print("🍽️ Загружаем модель...")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
model = joblib.load(MODEL_DIR / "model.pkl")
features = joblib.load(MODEL_DIR / "features.joblib")
print("✅ Модель готова!")

io_lock = Lock()


# ==== Главная / health ====
@app.get("/health")
def health():
    return {"status": "ok", "model": "food-recs"}


@app.get("/", include_in_schema=False)
def root():
    """Если рядом лежит index.html — отдать его, иначе простое сообщение."""
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Food API работает. Открой /docs для Swagger UI."}


# ==== Вспомогательная функция загрузки и чистки данных ====
def load_and_clean_food() -> pd.DataFrame:
    if not FOOD_DB.exists():
        raise HTTPException(500, f"Файл с блюдами не найден: {FOOD_DB}")

    try:
        df = pd.read_csv(FOOD_DB)
    except Exception as e:
        raise HTTPException(500, f"Не удалось прочитать {FOOD_DB}: {e}")

    required_cols = ["title", "rating", "calories", "protein", "fat", "sodium"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(500, f"В food_clean_example.csv нет колонок: {missing}")

    # Чистим числовые колонки
    num_cols = ["rating", "calories", "protein", "fat", "sodium"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["title"])
    if df.empty:
        raise HTTPException(500, "После чистки не осталось ни одного блюда")

    return df


# ==== Эндпоинт: просто посмотреть блюда ====
@app.get("/dishes")
def get_dishes(limit: int = Query(50, ge=1, le=1000)):
    df = load_and_clean_food()
    return df.tail(limit).to_dict(orient="records")


# ==== Эндпоинт: рекомендации ====
@app.get("/recommend")
def recommend(
    max_calories: float = Query(9999, ge=0, description="Максимум ккал"),
    min_protein: float = Query(0, ge=0, description="Минимум белка"),
    top_n: int = Query(10, ge=1, le=50, description="Сколько блюд вернуть"),
):
    """
    Возвращает top_n лучших блюд по модели
    с учётом max_calories и min_protein.
    """
    with io_lock:
        df = load_and_clean_food()

        # Фильтры диеты
        if max_calories < 9999:
            df = df[df["calories"] <= max_calories]
        if min_protein > 0:
            df = df[df["protein"] >= min_protein]

        if df.empty:
            raise HTTPException(404, "Ничего не найдено по заданным фильтрам")

        # Те же фичи, что при обучении модели
        df = df.copy()
        df["prot_kcal"] = df["protein"] / (df["calories"] + 0.001)
        df["fat_kcal"] = df["fat"] / (df["calories"] + 0.001)
        df["salt_kcal"] = df["sodium"] / (df["calories"] + 0.001)

        # Подготовка X
        try:
            X = df[features].values.astype(np.float32)
        except KeyError as e:
            raise HTTPException(
                500,
                f"Список features из модели не совпадает с колонками df. "
                f"features={features}, ошибка: {e}",
            )

        # Проверка NaN
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            raise HTTPException(500, f"Во входе модели есть NaN (кол-во: {nan_count})")

        # Предсказание
        try:
            X_sc = scaler.transform(X)
            probs = model.predict_proba(X_sc)[:, 1]
        except Exception as e:
            raise HTTPException(500, f"Ошибка при работе модели: {e}")

        df["score"] = probs
        top = df.sort_values("score", ascending=False).head(top_n)

        return {
            "count": int(len(top)),
            "filters": {
                "max_calories": max_calories,
                "min_protein": min_protein,
                "top_n": top_n,
            },
            "dishes": top[
                ["title", "rating", "calories", "protein", "fat", "sodium", "score"]
            ]
            .round(3)
            .to_dict(orient="records"),
        }


# ==== Эндпоинт: простая статистика ====
@app.get("/stats")
def get_stats():
    df = load_and_clean_food()
    return {
        "total_dishes": int(len(df)),
        "avg_rating": float(df["rating"].mean()),
        "avg_calories": float(df["calories"].mean()),
        "avg_protein": float(df["protein"].mean()),
    }


if __name__ == "__main__":
    import uvicorn
    DATA_DIR.mkdir(exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)

