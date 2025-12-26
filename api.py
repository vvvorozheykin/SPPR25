from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Request
import uvicorn
from datetime import datetime
from typing import Optional, List
import time

app = FastAPI(
    title="🍽️ БЖУ PRO v5.2",
    description="Умный анализатор БЖУ рационов с графиками",
    version="5.2.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 ПУТИ
BASE_DIR = Path(r"C:\Users\User\Desktop\🤖model")
DATA_PATH = Path(r"C:\Users\User\Desktop\data\food_clean_example.csv")
INDEX_PATH = BASE_DIR / "index.txt"

# 🗄️ КЭШ данных
data_cache = None
cache_time = 0
CACHE_DURATION = 300  # 5 минут


def load_data(force: bool = False):
    """🔄 Загрузка + автофикс CSV"""
    global data_cache, cache_time

    if not force and data_cache is not None and (time.time() - cache_time) < CACHE_DURATION:
        return data_cache.copy()

    try:
        # Читаем CSV
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            print(f"📊 Загружено {len(df)} строк из CSV")
        else:
            print("⚠️ CSV не найден — тестовые данные")
            df = pd.DataFrame({
                "title": ["Куриная грудка 100г", "Тунец в соку 100г", "Гречка вареная 100г",
                          "Творог 5% 100г", "Яйца вареные 2шт", "Треска запеченная 100г",
                          "Овсянка на воде 100г", "Кефир 1% 200мл", "Говядина отварная 100г"],
                "calories": [165, 116, 123, 121, 155, 82, 68, 38, 250],
                "protein": [31.0, 25.1, 4.0, 17.2, 13.0, 17.8, 2.5, 3.0, 26.7],
                "fat": [3.6, 1.3, 1.1, 5.0, 11.0, 0.7, 1.4, 1.0, 17.0],
                "sodium": [74, 353, 5, 36, 124, 65, 2, 42, 72],
                "rating": [4.8, 4.7, 4.2, 4.6, 4.3, 4.9, 4.1, 4.4, 4.5]
            })

        # ✅ АВТОФИКС колонок
        cols_needed = ["title", "calories", "protein", "fat", "carbs", "sodium", "rating"]
        for col in cols_needed:
            if col not in df.columns:
                if col == "title":
                    df[col] = [f"Блюдо {i + 1}" for i in range(len(df))]
                elif col == "carbs":
                    df[col] = np.maximum(0.0,
                                         (df["calories"] - df["protein"].fillna(0) * 4 - df["fat"].fillna(0) * 9) / 4)
                    print(f"🔧 Автосоздал carbs")
                elif col == "sodium":
                    df[col] = 50.0
                elif col == "rating":
                    df[col] = 4.5
                else:
                    df[col] = 0.0

        # ✅ Числовые типы
        num_cols = ["calories", "protein", "fat", "carbs", "sodium", "rating"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # ✅ Универсальная формула Score
        df["score"] = (
                df["protein"] * 0.4 +
                df["carbs"] * 0.2 +
                np.clip(1 - df["fat"] / 20, 0, 1) * 0.2 +
                (df["rating"] / 5) * 0.1 +
                np.log1p(df["calories"] / 100) * 0.1
        )

        data_cache = df
        cache_time = time.time()
        print(f"✅ Данные готовы: {len(df)} блюд | Score avg: {df['score'].mean():.1f}")
        return df.copy()

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return pd.DataFrame([{
            "title": "Тестовое блюдо", "calories": 100.0, "protein": 20.0,
            "fat": 5.0, "carbs": 10.0, "sodium": 50.0, "rating": 4.5, "score": 20.0
        }])


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    print(f"❌ Глобальная ошибка: {exc}")
    return JSONResponse(status_code=500, content={"error": "Серверная ошибка"})


# 🟢 1. HEALTH CHECK
@app.get("/health")
async def health():
    """🟢 Статус API"""
    df = load_data()
    return {
        "status": "🟢 OK",
        "timestamp": datetime.now().isoformat(),
        "dishes": len(df),
        "bju_avg": {
            "calories": round(float(df["calories"].mean()), 1),
            "protein": round(float(df["protein"].mean()), 1),
            "fat": round(float(df["fat"].mean()), 1),
            "carbs": round(float(df["carbs"].mean()), 1),
            "score": round(float(df["score"].mean()), 1)
        },
        "top3": df.nlargest(3, "score")[["title", "score"]].round(1).to_dict('records'),
        "csv_exists": DATA_PATH.exists()
    }


# 📊 2. ГРАФИК
@app.get("/chart")
async def chart(top_n: int = Query(5, ge=1, le=20)):
    """📊 Топ-N для Chart.js"""
    df = load_data()
    top = df.nlargest(top_n, "score")[["title", "score", "protein", "fat", "carbs", "calories", "rating"]].round(1)
    return {
        "success": True,
        "top_n": top_n,
        "dishes": top.to_dict('records'),
        "max_score": float(top["score"].max())
    }


# 🔍 3. РЕКОМЕНДАЦИИ
@app.get("/recommend")
async def recommend(
        max_cal: float = Query(9999, ge=0),
        min_prot: float = Query(0, ge=0),
        max_fat: float = Query(9999, ge=0),
        min_carbs: float = Query(0, ge=0),
        search: str = Query(""),
        sort: str = Query("score", pattern="^(score|protein|fat|carbs|calories|rating)$"),
        order: str = Query("desc", pattern="^(asc|desc)$"),
        limit: int = Query(10, ge=1, le=100)
):
    """🔍 Фильтрация БЖУ блюд"""
    df = load_data()

    # Фильтры
    mask = (
            (df["calories"] <= max_cal) &
            (df["protein"] >= min_prot) &
            (df["fat"] <= max_fat) &
            (df["carbs"] >= min_carbs)
    )

    result = df[mask].copy()

    if search.strip():
        result = result[result["title"].astype(str).str.contains(search.strip(), case=False, na=False, regex=False)]

    if len(result) == 0:
        return {"dishes": [], "count": 0, "message": "Блюда не найдены"}

    # Сортировка
    ascending = order == "asc"
    result = result.sort_values(sort, ascending=ascending).head(limit)

    cols = ["title", "calories", "protein", "fat", "carbs", "sodium", "rating", "score"]
    dishes = result[cols].round(1).to_dict('records')

    return {
        "dishes": dishes,
        "count": len(result),
        "filters": {"max_cal": max_cal, "min_prot": min_prot, "max_fat": max_fat, "min_carbs": min_carbs},
        "stats": {
            "protein_avg": round(float(result["protein"].mean()), 1),
            "fat_avg": round(float(result["fat"].mean()), 1),
            "carbs_avg": round(float(result["carbs"].mean()), 1),
            "score_avg": round(float(result["score"].mean()), 1)
        }
    }


# 🍽️ 4. ДНЕВНОЙ РАЦИОН — ✅ ПОЛНЫЙ ФИКС + УМНАЯ ОЦЕНКА
@app.get("/plan")
async def daily_plan(
        calories: int = Query(2000, ge=1000, le=5000),
        protein: int = Query(150, ge=50, le=400),
        fat: int = Query(70, ge=30, le=200),
        carbs: int = Query(250, ge=50, le=600),
        meals: int = Query(4, ge=2, le=6)
):
    """🍽️ Полный рацион — ФИКС UserWarning + НОВАЯ ОЦЕНКА"""
    df = load_data()

    # Нормы на прием
    norms = {
        "calories": calories / meals,
        "protein": protein / meals,
        "fat": fat / meals,
        "carbs": carbs / meals
    }

    # Score для подбора
    df["plan_score"] = (
            np.clip(df["protein"] / norms["protein"], 0, 2) * 0.4 +
            np.clip(df["carbs"] / norms["carbs"], 0, 2) * 0.3 +
            np.clip(1 - abs(df["fat"] - norms["fat"]) / max(norms["fat"] * 0.5, 1), 0, 1) * 0.2 +
            (df["rating"] / 5) * 0.1
    )

    # Формируем рацион
    plan = []
    totals = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    available = df.copy()
    meal_names = ["☕ Завтрак", "🍲 Обед", "🥩 Ужин", "🥜 Перекус 1", "🍎 Перекус 2", "🥛 Перекус 3"]

    for i in range(meals):
        cal_ok = available["calories"] <= norms["calories"] * 1.5
        prot_ok = available["protein"] >= norms["protein"] * 0.5
        candidates = available[cal_ok & prot_ok]

        if len(candidates) == 0:
            candidates = available.nlargest(10, "plan_score")

        best = candidates.nlargest(1, "plan_score").iloc[0]

        meal = {
            "meal": meal_names[i % len(meal_names)],
            "title": str(best["title"]),
            "calories": float(best["calories"]),
            "protein": float(best["protein"]),
            "fat": float(best["fat"]),
            "carbs": float(best["carbs"]),
            "score": float(best["plan_score"])
        }

        plan.append(meal)
        for key in totals:
            totals[key] += meal[key]

        # ✅ ФИКС: ТОЧНОЕ сравнение строк
        best_title = str(best["title"])
        available = available[~(available["title"].astype(str) == best_title)]

    # Проценты выполнения
    percents = {
        "calories": round((totals["calories"] / calories) * 100, 1),
        "protein": round((totals["protein"] / protein) * 100, 1),
        "fat": round((totals["fat"] / fat) * 100, 1),
        "carbs": round((totals["carbs"] / carbs) * 100, 1)
    }

    # 🧠 УМНАЯ ОЦЕНКА КАЧЕСТВА (0-100)
    quality_score = 0

    # 1. Точность % (50%)
    for pct in percents.values():
        deviation = abs(pct - 100)
        if deviation <= 5:
            quality_score += 12.5  # Идеально
        elif deviation <= 15:
            quality_score += 10  # Хорошо
        elif deviation <= 30:
            quality_score += 7.5  # Нормально
        else:
            quality_score += 5  # Плохо

    # 2. Баланс БЖУ (20%)
    prot_carb_diff = abs(percents["protein"] - percents["carbs"])
    if prot_carb_diff <= 10:
        quality_score += 20
    elif prot_carb_diff <= 25:
        quality_score += 15
    else:
        quality_score += 10

    # 3. Средний score блюд (20%)
    avg_meal_score = np.mean([m["score"] for m in plan])
    quality_score += min(avg_meal_score * 4, 20)

    # 4. Разнообразие калорий (10%)
    cal_std = np.std([m["calories"] for m in plan])
    diversity = min(cal_std / norms["calories"] * 5, 10)
    quality_score += diversity

    quality_score = round(max(0, min(quality_score, 100)), 1)

    return {
        "success": True,
        "plan": plan,
        "totals": totals,
        "percents": percents,
        "targets": {"calories": calories, "protein": protein, "fat": fat, "carbs": carbs, "meals": meals},
        "quality": quality_score,
        "quality_breakdown": {
            "percent_accuracy": round(
                sum([12.5 if abs(p - 100) <= 5 else 10 if abs(p - 100) <= 15 else 7.5 if abs(p - 100) <= 30 else 5 for p
                     in percents.values()]), 1),
            "bju_balance": 20 - abs(percents["protein"] - percents["carbs"]) * 0.5,
            "meal_scores": round(avg_meal_score * 4, 1),
            "diversity": round(diversity, 1)
        }
    }


# 📈 5. СТАТИСТИКА
@app.get("/stats")
async def stats():
    """📈 Полная статистика"""
    df = load_data()

    bju_total = float(df["protein"].sum() + df["fat"].sum() + df["carbs"].sum())
    bju_dist = {
        "protein_pct": round((df["protein"].sum() / bju_total) * 100, 1),
        "fat_pct": round((df["fat"].sum() / bju_total) * 100, 1),
        "carbs_pct": round((df["carbs"].sum() / bju_total) * 100, 1)
    }

    metrics = {}
    for col in ["calories", "protein", "fat", "carbs", "score"]:
        data = df[col]
        metrics[col] = {
            "avg": round(float(data.mean()), 1),
            "min": round(float(data.min()), 1),
            "max": round(float(data.max()), 1),
            "p95": round(float(data.quantile(0.95)), 1)
        }

    top5 = df.nlargest(5, "score")[["title", "score", "protein", "fat", "carbs"]].round(1)

    return {
        "total_dishes": len(df),
        "bju_distribution": bju_dist,
        "metrics": metrics,
        "top5": top5.to_dict('records'),
        "healthy": len(df[(df["protein"] >= 20) & (df["fat"] <= 15)]),
        "avg_quality": round(float(df["score"].mean()), 1)
    }


# 🏠 6. ГЛАВНАЯ СТРАНИЦА
@app.get("/", include_in_schema=False)
async def root():
    """📱 Главная с index.txt или fallback"""
    try:
        if INDEX_PATH.exists():
            with open(INDEX_PATH, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        print(f"❌ index.txt ошибка: {e}")

    return HTMLResponse("""
<!DOCTYPE html>
<html><head><title>🍽️ БЖУ PRO v5.2</title>
<style>body{font-family:Arial;padding:40px;background:#f5f5f5;text-align:center;}
h1{color:#10b981;font-size:3em;}a{background:#10b981;color:white;padding:15px 30px;border-radius:10px;text-decoration:none;font-weight:bold;display:inline-block;margin:10px;font-size:16px;}
a:hover{background:#059669;transform:translateY(-2px);transition:0.3s;}</style></head>
<body>
<h1>🍽️ БЖУ PRO v5.2 ✅</h1>
<p><strong>СОЗДАЙТЕ ФАЙЛ:</strong><br>C:\\Users\\User\\Desktop\\🤖model\\index.txt</p>
<p>
<a href="/docs">📚 API Документация (Swagger)</a><br>
<a href="/redoc">📖 API Документация (ReDoc)</a><br>
<a href="/health">✅ Тест API</a><br>
<a href="/chart?top_n=5">📊 График Топ-5</a><br>
<a href="/recommend?min_prot=20&limit=10">🔍 Протеин 20г+</a><br>
<a href="/plan?calories=2000&meals=4">🍽️ Рацион 2000 ккал</a><br>
<a href="/stats">📈 Статистика БЖУ</a>
</p>
</body></html>
    """)

if __name__ == "__main__":
    print("=" * 80)
    print("🍽️ БЖУ PRO v5.2 — ПОЛНЫЙ API!")
    print("📁 CSV данные:", DATA_PATH)
    print("📁 index.txt:  ", INDEX_PATH)
    print("\n🚀 ОСНОВНЫЕ ССЫЛКИ:")
    print("📱 http://localhost:8000/")
    print("📚 http://localhost:8000/docs")
    print("📊 http://localhost:8000/chart?top_n=5")
    print("🔍 http://localhost:8000/recommend?min_prot=20")
    print("🍽️ http://localhost:8000/plan?calories=2000&meals=4")
    print("📈 http://localhost:8000/stats")
    print("=" * 80)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
