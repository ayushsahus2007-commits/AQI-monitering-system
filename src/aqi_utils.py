from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"]

AQI_BANDS = [
    {
        "label": "Good",
        "upper": 50,
        "color": "#2e8b57",
        "accent": "#cdeed9",
        "advisory": "Air quality is satisfactory. Outdoor activity is generally safe.",
        "tip": "Keep tracking emissions so the city stays in the safe zone.",
    },
    {
        "label": "Satisfactory",
        "upper": 100,
        "color": "#87a330",
        "accent": "#e8efc5",
        "advisory": "Minor breathing discomfort may affect sensitive groups.",
        "tip": "People with asthma should carry medication during longer outdoor exposure.",
    },
    {
        "label": "Moderate",
        "upper": 200,
        "color": "#d28b17",
        "accent": "#f7e0b6",
        "advisory": "Breathing discomfort can occur for people with lung disease.",
        "tip": "Reduce outdoor exertion during peak traffic or industrial hours.",
    },
    {
        "label": "Poor",
        "upper": 300,
        "color": "#cc5a2f",
        "accent": "#f5c8b6",
        "advisory": "Breathing discomfort is possible for most people after exposure.",
        "tip": "Limit prolonged outdoor activity and use a mask in crowded areas.",
    },
    {
        "label": "Very Poor",
        "upper": 400,
        "color": "#9a3d3d",
        "accent": "#e8c0c0",
        "advisory": "Respiratory illness risk increases on prolonged exposure.",
        "tip": "Avoid strenuous outdoor work and improve ventilation filtration indoors.",
    },
    {
        "label": "Severe",
        "upper": float("inf"),
        "color": "#5a2a2a",
        "accent": "#ddb8b8",
        "advisory": "Health effects can impact everyone, including healthy adults.",
        "tip": "Minimize outdoor exposure and activate emergency air-quality precautions.",
    },
]

CATEGORY_DETAILS = {band["label"]: band for band in AQI_BANDS}


def get_aqi_band(aqi: float) -> dict:
    value = float(aqi)
    for band in AQI_BANDS:
        if value <= band["upper"]:
            return band
    return AQI_BANDS[-1]


def get_aqi_category(aqi: float) -> str:
    return get_aqi_band(aqi)["label"]


def get_health_advisory(category: str) -> str:
    band = CATEGORY_DETAILS.get(category, CATEGORY_DETAILS["Good"])
    return band["advisory"]


def get_aqi_color(aqi: float) -> str:
    return get_aqi_band(aqi)["color"]


def get_aqi_accent(aqi: float) -> str:
    return get_aqi_band(aqi)["accent"]


def get_response_tip(aqi: float) -> str:
    return get_aqi_band(aqi)["tip"]


def build_feature_frame(feature_values: Mapping[str, float]) -> pd.DataFrame:
    row = {feature: float(feature_values.get(feature, 0.0)) for feature in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def estimate_prediction_interval(model, feature_frame: pd.DataFrame) -> tuple[float | None, float | None, float]:
    if not hasattr(model, "estimators_"):
        return None, None, 0.0

    feature_array = feature_frame.to_numpy()
    tree_predictions = np.array(
        [float(estimator.predict(feature_array)[0]) for estimator in model.estimators_],
        dtype=float,
    )
    low = float(np.percentile(tree_predictions, 10))
    high = float(np.percentile(tree_predictions, 90))
    spread = float(tree_predictions.std())
    return low, high, spread


def get_confidence_label(spread: float) -> str:
    if spread <= 15:
        return "Stable"
    if spread <= 35:
        return "Moderate spread"
    return "High spread"


def predict_aqi(model, feature_values: Mapping[str, float]) -> dict:
    feature_frame = build_feature_frame(feature_values)
    prediction = float(model.predict(feature_frame)[0])
    interval_low, interval_high, spread = estimate_prediction_interval(model, feature_frame)
    band = get_aqi_band(prediction)

    return {
        "aqi": prediction,
        "category": band["label"],
        "color": band["color"],
        "accent": band["accent"],
        "advisory": band["advisory"],
        "tip": band["tip"],
        "interval_low": interval_low,
        "interval_high": interval_high,
        "spread": spread,
        "confidence": get_confidence_label(spread),
    }


def build_city_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {feature: "mean" for feature in FEATURE_COLUMNS}
    aggregations["AQI"] = "mean"

    summary = df.groupby(["City", "Year"], as_index=False).agg(aggregations)
    samples = df.groupby(["City", "Year"]).size().reset_index(name="Samples")
    return summary.merge(samples, on=["City", "Year"], how="left")


def get_reference_profile(summary_df: pd.DataFrame, city: str, year: int) -> dict:
    row = summary_df[(summary_df["City"] == city) & (summary_df["Year"] == year)]
    if row.empty:
        return {feature: 0.0 for feature in FEATURE_COLUMNS}
    return row.iloc[0][FEATURE_COLUMNS].to_dict()


def describe_pollutant_load(
    feature_values: Mapping[str, float],
    baseline_values: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_COLUMNS:
        current_value = float(feature_values.get(feature, 0.0))
        baseline_value = float(baseline_values.get(feature, 0.0))
        safe_baseline = baseline_value if baseline_value > 0 else 1.0
        rows.append(
            {
                "Pollutant": feature,
                "Value": current_value,
                "Baseline": baseline_value,
                "Relative Load": current_value / safe_baseline,
                "Delta": current_value - baseline_value,
            }
        )
    return pd.DataFrame(rows).sort_values("Relative Load", ascending=False)


def simulate_reduction(
    model,
    feature_values: Mapping[str, float],
    pollutant: str,
    reduction_pct: float,
) -> dict:
    adjusted_values = {feature: float(feature_values.get(feature, 0.0)) for feature in FEATURE_COLUMNS}
    reduction_factor = max(0.0, 1.0 - (reduction_pct / 100.0))
    adjusted_values[pollutant] = adjusted_values[pollutant] * reduction_factor
    result = predict_aqi(model, adjusted_values)
    result["pollutant"] = pollutant
    result["reduction_pct"] = reduction_pct
    result["inputs"] = adjusted_values
    return result


def benchmark_model(df: pd.DataFrame, random_state: int = 42) -> dict:
    features = df[FEATURE_COLUMNS]
    target = df["AQI"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    benchmark = RandomForestRegressor(
        n_estimators=120,
        random_state=random_state,
        n_jobs=-1,
    )
    benchmark.fit(x_train, y_train)
    predictions = benchmark.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, predictions)),
        "feature_importance": pd.DataFrame(
            {
                "Pollutant": FEATURE_COLUMNS,
                "Importance": benchmark.feature_importances_,
            }
        ).sort_values("Importance", ascending=False),
    }
