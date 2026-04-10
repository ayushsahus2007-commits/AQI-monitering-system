# Air Quality Index (AQI) Monitoring System v2

An upgraded machine learning dashboard that predicts Air Quality Index (AQI), benchmarks city pollution trends, and simulates pollutant-reduction scenarios for final project review demonstrations.

## What Is New In v2

- Redesigned Streamlit dashboard with a cleaner review-ready interface
- City intelligence section with AQI rank, year-over-year change, and same-year leaderboard
- Interactive prediction lab with live pollutant inputs
- Prediction confidence band using Random Forest ensemble spread
- Mitigation simulator to estimate AQI improvement after reducing major pollutants
- Model analytics section with RMSE, R², feature importance, and pollutant correlation views
- Refactored AQI utility layer for cleaner reusable logic

## Project Objective

This project uses historical pollutant data from Indian cities to:

- Predict AQI using a machine learning model
- Classify air quality into standard AQI categories
- Provide health advisory guidance
- Compare pollution trends across cities and years
- Support decision-making through analytics and scenario testing

## Core Features

### 1. City Intelligence Dashboard
- Select a city and year
- View average AQI and AQI category
- Compare against national city average
- Track historical AQI trends
- Benchmark against another city
- View most polluted and cleanest cities for the selected year

### 2. Prediction Lab
- Enter pollutant values for PM2.5, PM10, NO2, NH3, CO, SO2, and O3
- Generate AQI prediction instantly
- View AQI category and health advisory
- Inspect prediction confidence range
- Compare current inputs against the selected city baseline

### 3. Mitigation Simulator
- Reduce one pollutant at a time by a chosen percentage
- Estimate how much AQI improves after intervention
- Identify the strongest pollutant target for action

### 4. Model Analytics
- Benchmark Random Forest performance on train-test split
- View feature importance scores
- Inspect pollutant correlation with AQI
- Analyze AQI category distribution across city-year records

## AQI Category Standard

| AQI Range | Category |
| --- | --- |
| 0-50 | Good |
| 51-100 | Satisfactory |
| 101-200 | Moderate |
| 201-300 | Poor |
| 301-400 | Very Poor |
| 401+ | Severe |

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

## Project Structure

```text
AQI-monitering-system/
├── App/
│   └── app.py
├── Data/
│   └── cleaned_air_quality.csv
├── Docs/
│   ├── abstract.docx
│   ├── project_report.docx
│   ├── ppt.pptx
│   └── viva_notes.txt
├── Model/
│   └── aqi_model.pkl
├── Notebooks/
│   ├── analysis.ipynb
│   ├── data_cleaning.ipynb
│   ├── evaluation.ipynb
│   └── model_training.ipynb
├── src/
│   └── aqi_utils.py
├── requirements.txt
└── streamlit_app.py
```

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

3. Open the local Streamlit URL in the browser.

## Model Inputs

The prediction model uses these pollutant features:

- PM2.5
- PM10
- NO2
- NH3
- CO
- SO2
- O3

## Review Talking Points

If you want strong presentation points for viva or project review, focus on these:

- The app is no longer only a predictor; it now behaves like a decision-support dashboard.
- The system combines forecasting, benchmarking, and mitigation analysis in one interface.
- Confidence range adds transparency to the machine learning prediction.
- Feature importance and correlation plots help explain why the model behaves the way it does.
- Scenario simulation makes the project more practical for environmental planning discussions.

## Team

- Ayush: Team Lead and ML Model
- Avinash: Data Cleaning
- Rohith: Analysis and Insights
- Hiten: Web Application
- Rasika: AQI Logic and Integration
