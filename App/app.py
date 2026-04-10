from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "cleaned_air_quality.csv"
MODEL_PATH = BASE_DIR / "Model" / "aqi_model.pkl"
sys.path.append(str(BASE_DIR))

from src.aqi_utils import build_city_year_summary, get_reference_profile, predict_aqi


def main():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    df = pd.read_csv(DATA_PATH)
    summary = build_city_year_summary(df)
    profile = get_reference_profile(summary, "Delhi", 2025)
    model = joblib.load(MODEL_PATH)
    result = predict_aqi(model, profile)

    print("AQI Monitoring System CLI smoke test")
    print(f"Sample city-year: Delhi 2025")
    print(f"Predicted AQI: {result['aqi']:.2f}")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']}")


if __name__ == "__main__":
    main()
