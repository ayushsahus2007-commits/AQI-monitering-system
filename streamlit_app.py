import os
import sys
import warnings

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

st.set_page_config(
    page_title="AQI Monitoring System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.append(os.path.abspath("."))

from src.aqi_utils import (
    FEATURE_COLUMNS,
    benchmark_model,
    build_city_year_summary,
    describe_pollutant_load,
    get_aqi_category,
    get_aqi_color,
    get_health_advisory,
    get_reference_profile,
    get_response_tip,
    predict_aqi,
    simulate_reduction,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "aqi_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Data", "cleaned_air_quality.csv")


@st.cache_resource
def load_model():
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def prepare_summary(df: pd.DataFrame):
    return build_city_year_summary(df)


@st.cache_data
def prepare_benchmark(df: pd.DataFrame):
    return benchmark_model(df)


def inject_styles():
    st.markdown(
        """
        <style>
            :root {
                --forest: #335c4b;
                --moss: #8aa37b;
                --clay: #c76d3c;
                --sand: #f6efe3;
                --ink: #22302a;
                --muted: #61746c;
            }
            .stApp {
                background:
                    radial-gradient(circle at 14% 2%, rgba(231, 128, 71, 0.23), transparent 28%),
                    radial-gradient(circle at 88% 6%, rgba(51, 92, 75, 0.22), transparent 30%),
                    radial-gradient(circle at 50% 120%, rgba(80, 123, 103, 0.13), transparent 45%),
                    linear-gradient(180deg, #fcf7ef 0%, #f5ede0 100%);
                color: var(--ink);
                font-family: "Trebuchet MS", "Avenir Next", sans-serif;
            }
            h1, h2, h3 {
                color: var(--ink);
                font-family: "Baskerville", "Palatino Linotype", serif;
                letter-spacing: 0.02em;
            }
            .hero-shell {
                background: linear-gradient(135deg, rgba(51, 92, 75, 0.95), rgba(26, 42, 34, 0.90));
                border-radius: 22px;
                color: #f8f3ea;
                padding: 28px 30px;
                box-shadow: 0 20px 45px rgba(34, 48, 42, 0.12);
                margin-bottom: 1rem;
                transition: transform 0.35s ease, box-shadow 0.35s ease;
                animation: fadeSlideIn 0.8s ease forwards;
            }
            .hero-shell:hover {
                transform: translateY(-4px);
                box-shadow: 0 26px 54px rgba(34, 48, 42, 0.2);
            }
            .hero-kicker {
                color: #d6e7dd;
                font-size: 0.85rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }
            .hero-title {
                font-size: 2.3rem;
                font-weight: 600;
                margin-bottom: 0.55rem;
            }
            .hero-copy {
                color: #e7efe9;
                max-width: 850px;
                line-height: 1.6;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.78);
                border-radius: 18px;
                border-top: 5px solid var(--forest);
                padding: 16px 18px;
                min-height: 136px;
                box-shadow: 0 10px 30px rgba(73, 77, 71, 0.08);
                backdrop-filter: blur(4px);
                transition: transform 0.28s ease, box-shadow 0.28s ease, border-top-color 0.28s ease;
            }
            .stat-card:hover {
                transform: translateY(-6px) scale(1.015);
                box-shadow: 0 20px 42px rgba(73, 77, 71, 0.18);
            }
            .stat-label {
                color: var(--muted);
                font-size: 0.88rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.4rem;
            }
            .stat-value {
                color: var(--ink);
                font-size: 2rem;
                font-weight: 700;
                line-height: 1.1;
                margin-bottom: 0.35rem;
            }
            .stat-subtitle {
                color: var(--muted);
                line-height: 1.45;
                font-size: 0.92rem;
            }
            .panel-card {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 10px 30px rgba(73, 77, 71, 0.08);
                margin-bottom: 1rem;
                border: 1px solid rgba(51, 92, 75, 0.08);
                transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            }
            .panel-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 18px 34px rgba(73, 77, 71, 0.15);
                border-color: rgba(51, 92, 75, 0.22);
            }
            .insight-card {
                background: #fffdfa;
                border-left: 5px solid var(--clay);
                border-radius: 14px;
                padding: 14px 16px;
                margin-bottom: 0.8rem;
                box-shadow: 0 8px 22px rgba(73, 77, 71, 0.06);
                transition: transform 0.26s ease, box-shadow 0.26s ease;
            }
            .insight-card:hover {
                transform: translateX(3px);
                box-shadow: 0 12px 28px rgba(73, 77, 71, 0.12);
            }
            .prediction-card {
                border-radius: 20px;
                padding: 22px;
                color: #1f2522;
                box-shadow: 0 14px 35px rgba(73, 77, 71, 0.1);
                transition: transform 0.28s ease, box-shadow 0.28s ease;
            }
            .prediction-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 18px 42px rgba(73, 77, 71, 0.16);
            }
            .section-caption {
                color: var(--muted);
                margin-top: -0.2rem;
                margin-bottom: 1rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(51, 92, 75, 0.08);
                padding: 12px 14px;
                border-radius: 16px;
                transition: transform 0.24s ease, box-shadow 0.24s ease;
            }
            div[data-testid="stMetric"]:hover {
                transform: translateY(-3px);
                box-shadow: 0 10px 24px rgba(73, 77, 71, 0.12);
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f6efe3 0%, #efe5d3 100%);
            }
            div[data-testid="stVegaLiteChart"] {
                border-radius: 14px;
                border: 1px solid rgba(51, 92, 75, 0.1);
                background: rgba(255, 253, 250, 0.88);
                padding: 8px 10px 2px 10px;
            }
            @keyframes fadeSlideIn {
                0% { opacity: 0; transform: translateY(14px); }
                100% { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(title: str, value: str, subtitle: str, accent: str = "#335c4b"):
    st.markdown(
        f"""
        <div class="stat-card" style="border-top-color: {accent};">
            <div class="stat-label">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(message: str):
    st.markdown(f"<div class='insight-card'>{message}</div>", unsafe_allow_html=True)


def build_future_watchlist(summary_df: pd.DataFrame) -> pd.DataFrame:
    if 2025 not in set(summary_df["Year"].tolist()):
        return pd.DataFrame(columns=["City", "AQI 2025", "Change vs 2024", "Risk"])

    current = summary_df[summary_df["Year"] == 2025][["City", "AQI"]].rename(columns={"AQI": "AQI 2025"})
    previous = summary_df[summary_df["Year"] == 2024][["City", "AQI"]].rename(columns={"AQI": "AQI 2024"})
    merged = current.merge(previous, on="City", how="left")
    merged["Change vs 2024"] = merged["AQI 2025"] - merged["AQI 2024"]

    def risk_label(value: float) -> str:
        if value <= 100:
            return "Low"
        if value <= 200:
            return "Medium"
        if value <= 300:
            return "High"
        return "Critical"

    merged["Risk"] = merged["AQI 2025"].apply(risk_label)
    return merged.sort_values("AQI 2025", ascending=False).reset_index(drop=True)


def evaluate_exposure_plan(
    aqi: float,
    activity: str,
    outdoor_hours: float,
    sensitive_group: bool,
    mask_used: bool,
) -> dict:
    activity_factor = {
        "Daily commute": 1.0,
        "Walking": 1.1,
        "Cycling": 1.25,
        "Outdoor sports": 1.4,
        "Outdoor work": 1.55,
    }.get(activity, 1.0)

    score = aqi * outdoor_hours * activity_factor
    if sensitive_group:
        score *= 1.25
    if mask_used:
        score *= 0.86

    if score <= 130:
        return {
            "label": "Low exposure risk",
            "color": "#2e8b57",
            "message": "Routine outdoor plans are generally safe with normal care.",
            "steps": [
                "Keep hydration and follow normal commute plans.",
                "Track AQI updates once or twice during the day.",
            ],
        }
    if score <= 260:
        return {
            "label": "Moderate exposure risk",
            "color": "#d28b17",
            "message": "Outdoor exposure should be managed with time and activity control.",
            "steps": [
                "Reduce high-intensity outdoor activity duration.",
                "Prefer cleaner travel windows and carry a quality mask.",
            ],
        }
    if score <= 420:
        return {
            "label": "High exposure risk",
            "color": "#cc5a2f",
            "message": "Limit prolonged outdoor time and protect vulnerable people.",
            "steps": [
                "Shift exercise or heavy activity indoors.",
                "Use masks consistently and keep indoor air filtration active.",
            ],
        }
    return {
        "label": "Critical exposure risk",
        "color": "#9a3d3d",
        "message": "Outdoor exposure should be minimized as much as possible.",
        "steps": [
            "Avoid non-essential outdoor trips, especially for sensitive groups.",
            "Follow emergency-level air quality precautions and indoor protection.",
        ],
    }


def format_delta(delta: float | None, lower_is_better: bool = True) -> str:
    if delta is None:
        return "No previous-year record"
    direction = "improved" if (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better) else "increased"
    return f"{delta:+.1f} AQI ({direction})"


def build_trend_chart(summary_df: pd.DataFrame, selected_city: str, compare_city: str | None):
    national_trend = summary_df.groupby("Year", as_index=False)["AQI"].mean()
    city_trend = summary_df[summary_df["City"] == selected_city].sort_values("Year")
    lines = [
        city_trend.assign(Series=selected_city),
        national_trend.assign(Series="National city average"),
    ]
    color_domain = [selected_city, "National city average"]
    color_range = ["#c76d3c", "#335c4b"]

    if compare_city:
        compare_trend = summary_df[summary_df["City"] == compare_city].sort_values("Year")
        lines.append(compare_trend.assign(Series=compare_city))
        color_domain.append(compare_city)
        color_range.append("#8aa37b")

    chart_df = pd.concat(lines, ignore_index=True)
    hover = alt.selection_point(fields=["Year"], nearest=True, on="pointerover", empty=False)

    line = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Year:Q", title="Year", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y("AQI:Q", title="Average AQI"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Year:Q", format=".0f"),
                alt.Tooltip("AQI:Q", format=".1f"),
            ],
        )
    )
    points = line.mark_circle(size=85).transform_filter(hover)
    rules = (
        alt.Chart(chart_df)
        .mark_rule(color="#44514b", strokeDash=[4, 3], opacity=0.4)
        .encode(x=alt.X("Year:Q", axis=alt.Axis(format="d", tickMinStep=1)))
        .transform_filter(hover)
    )
    return (line + points + rules).add_params(hover).properties(height=340, title="Historical AQI trajectory")


def build_pollutant_profile_chart(selected_row: pd.Series, year_frame: pd.DataFrame):
    national_profile = year_frame[FEATURE_COLUMNS].mean()
    profile_df = pd.DataFrame(
        {
            "Pollutant": FEATURE_COLUMNS,
            "Relative Level": [selected_row[feature] / max(national_profile[feature], 1.0) for feature in FEATURE_COLUMNS],
        }
    ).sort_values("Relative Level")
    profile_df["Signal"] = profile_df["Relative Level"].apply(lambda value: "Above baseline" if value > 1 else "Below baseline")

    bars = (
        alt.Chart(profile_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Pollutant:N", sort=profile_df["Pollutant"].tolist(), title=""),
            x=alt.X("Relative Level:Q", title="Relative intensity"),
            color=alt.Color(
                "Signal:N",
                scale=alt.Scale(domain=["Above baseline", "Below baseline"], range=["#c76d3c", "#8aa37b"]),
                legend=alt.Legend(title="Level"),
            ),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Relative Level:Q", format=".2f"),
                alt.Tooltip("Signal:N"),
            ],
        )
    )
    rule = alt.Chart(pd.DataFrame({"Benchmark": [1.0]})).mark_rule(color="#335c4b", strokeDash=[7, 4]).encode(x="Benchmark:Q")
    chart = (bars + rule).properties(height=300, title="Pollutant fingerprint vs same-year national profile")
    return chart, profile_df.sort_values("Relative Level", ascending=False)


def build_level_chart(load_df: pd.DataFrame):
    chart_df = load_df.sort_values("Value")
    pollutant_order = chart_df["Pollutant"].tolist()
    points_df = chart_df.melt(
        id_vars="Pollutant",
        value_vars=["Baseline", "Value"],
        var_name="Type",
        value_name="Concentration",
    )
    lines = (
        alt.Chart(chart_df)
        .mark_rule(color="#d4ddd8", strokeWidth=3)
        .encode(
            y=alt.Y("Pollutant:N", sort=pollutant_order, title=""),
            x=alt.X("Baseline:Q", title="Concentration"),
            x2=alt.X2("Value:Q"),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Baseline:Q", format=".2f"),
                alt.Tooltip("Value:Q", format=".2f"),
            ],
        )
    )
    points = (
        alt.Chart(points_df)
        .mark_circle(size=120)
        .encode(
            y=alt.Y("Pollutant:N", sort=pollutant_order, title=""),
            x=alt.X("Concentration:Q", title="Concentration"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=["Baseline", "Value"], range=["#8aa37b", "#c76d3c"]),
                legend=alt.Legend(title="Profile"),
            ),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Type:N"),
                alt.Tooltip("Concentration:Q", format=".2f"),
            ],
        )
    )
    return (lines + points).properties(height=320, title="Current pollutant inputs vs selected-city baseline")


def build_importance_chart(importance_df: pd.DataFrame):
    chart_df = importance_df.sort_values("Importance")
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=5, color="#335c4b")
        .encode(
            y=alt.Y("Pollutant:N", sort=chart_df["Pollutant"].tolist(), title=""),
            x=alt.X("Importance:Q", title="Importance score"),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Importance:Q", format=".4f"),
            ],
        )
        .properties(height=300, title="Model feature importance")
    )


def build_correlation_chart(df: pd.DataFrame):
    corr_df = (
        df[FEATURE_COLUMNS + ["AQI"]]
        .corr(numeric_only=True)["AQI"]
        .drop("AQI")
        .sort_values()
        .reset_index()
        .rename(columns={"index": "Pollutant", "AQI": "Correlation"})
    )

    return (
        alt.Chart(corr_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            y=alt.Y("Pollutant:N", sort=corr_df["Pollutant"].tolist(), title=""),
            x=alt.X("Correlation:Q", title="Pearson correlation"),
            color=alt.condition(
                alt.datum.Correlation > 0,
                alt.value("#c76d3c"),
                alt.value("#8aa37b"),
            ),
            tooltip=[
                alt.Tooltip("Pollutant:N"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(height=300, title="Correlation of pollutants with AQI")
    )


def build_category_distribution(summary_df: pd.DataFrame):
    dist_df = summary_df.copy()
    dist_df["Category"] = dist_df["AQI"].apply(get_aqi_category)
    ordered_categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    distribution = dist_df["Category"].value_counts().reindex(ordered_categories, fill_value=0)
    counts = pd.DataFrame(
        {
            "Category": distribution.index.tolist(),
            "Count": distribution.values.tolist(),
        }
    )

    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Category:N", sort=ordered_categories, axis=alt.Axis(labelAngle=20)),
            y=alt.Y("Count:Q", title="Number of city-year records"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=ordered_categories,
                    range=["#2e8b57", "#87a330", "#d28b17", "#cc5a2f", "#9a3d3d", "#5a2a2a"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Count:Q", format=".0f"),
            ],
        )
        .properties(height=280, title="Distribution of city-year AQI categories")
    )


def render_prediction_card(prediction: dict):
    st.markdown(
        f"""
        <div class="prediction-card" style="background: linear-gradient(135deg, {prediction['accent']}, #fffdfa); border-left: 8px solid {prediction['color']};">
            <div class="stat-label" style="color: #45544d;">Predicted AQI</div>
            <div style="font-size: 3rem; font-weight: 700; color: {prediction['color']}; line-height: 1;">{prediction['aqi']:.1f}</div>
            <div style="font-size: 1.1rem; font-weight: 700; margin-top: 0.35rem;">{prediction['category']}</div>
            <div style="margin-top: 0.8rem; line-height: 1.55;">{prediction['advisory']}</div>
            <div style="margin-top: 0.65rem; color: #45544d;">{prediction['tip']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

model = load_model()
df = load_dataset()
summary_df = prepare_summary(df)
benchmark = prepare_benchmark(df)

cities = sorted(summary_df["City"].unique().tolist())
years = sorted(summary_df["Year"].unique().tolist())
latest_year = int(max(years))

st.sidebar.markdown("## AQI Control Room")
st.sidebar.write("Use these filters to inspect city trends, benchmark performance, and test mitigation scenarios.")

default_city_index = cities.index("Delhi") if "Delhi" in cities else 0
selected_city = st.sidebar.selectbox("Select city", cities, index=default_city_index)
selected_year = st.sidebar.selectbox("Select year", years, index=len(years) - 1)
compare_options = ["None"] + [city for city in cities if city != selected_city]
compare_city = st.sidebar.selectbox("Compare against", compare_options)
compare_city = None if compare_city == "None" else compare_city
scenario_reduction = st.sidebar.slider("Scenario reduction (%)", min_value=5, max_value=40, value=15, step=5)

st.sidebar.markdown("---")
st.sidebar.metric("Dataset rows", f"{len(df):,}")
st.sidebar.metric("Tracked cities", f"{summary_df['City'].nunique()}")
st.sidebar.metric("Years covered", f"{min(years)}-{max(years)}")

selected_row = summary_df[(summary_df["City"] == selected_city) & (summary_df["Year"] == selected_year)].iloc[0]
year_frame = summary_df[summary_df["Year"] == selected_year].sort_values("AQI", ascending=False).reset_index(drop=True)
national_avg = float(year_frame["AQI"].mean())
rank = int(year_frame.index[year_frame["City"] == selected_city][0]) + 1

previous_year_row = summary_df[
    (summary_df["City"] == selected_city) & (summary_df["Year"] == selected_year - 1)
]
previous_aqi = float(previous_year_row.iloc[0]["AQI"]) if not previous_year_row.empty else None
yoy_delta = float(selected_row["AQI"] - previous_aqi) if previous_aqi is not None else None
national_gap = float(selected_row["AQI"] - national_avg)
selected_color = get_aqi_color(selected_row["AQI"])

compare_row = None
compare_gap = None
if compare_city:
    compare_row = summary_df[(summary_df["City"] == compare_city) & (summary_df["Year"] == selected_year)]
    if not compare_row.empty:
        compare_row = compare_row.iloc[0]
        compare_gap = float(selected_row["AQI"] - compare_row["AQI"])

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-kicker">Practical Air Quality Intelligence</div>
        <div class="hero-title">AQI Monitoring System</div>
        <div class="hero-copy">
            A polished forecasting and analytics workspace for Indian city air quality data.
            The platform combines benchmarking, prediction confidence, mitigation simulation, and practical
            exposure planning so the project behaves like a decision-support system, not just a predictor.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(4)
with hero_cols[0]:
    render_stat_card(
        "Selected AQI",
        f"{selected_row['AQI']:.1f}",
        f"{selected_city} in {selected_year} falls in the {get_aqi_category(selected_row['AQI'])} band.",
        selected_color,
    )
with hero_cols[1]:
    render_stat_card(
        "Year-over-year change",
        format_delta(yoy_delta),
        "Negative change means air quality improved compared with the previous year.",
        "#c76d3c",
    )
with hero_cols[2]:
    render_stat_card(
        "National rank",
        f"#{rank} / {len(year_frame)}",
        f"{selected_city} is {'above' if national_gap > 0 else 'below'} the same-year national city average by {abs(national_gap):.1f} AQI.",
        "#8aa37b",
    )
with hero_cols[3]:
    compare_subtitle = (
        f"{selected_city} is {abs(compare_gap):.1f} AQI {'higher' if compare_gap > 0 else 'lower'} than {compare_city}."
        if compare_gap is not None
        else "Choose a compare city in the sidebar to enable peer benchmarking."
    )
    render_stat_card("Peer benchmark", compare_city or "Not set", compare_subtitle, "#aa8f5d")

with st.expander("System highlights"):
    st.write(
        """
        - Multi-section dashboard with city intelligence, live prediction lab, and model analytics.
        - Peer benchmarking using national average, city ranks, and side-by-side trend comparison.
        - Prediction confidence band derived from the Random Forest ensemble spread.
        - Mitigation simulator that estimates AQI improvement after pollutant reduction actions.
        - Practical daily exposure planner for real-life decisions.
        - Expanded dataset coverage from 2015 to 2025.
        """
    )

overview_tab, prediction_tab, analytics_tab = st.tabs(
    ["City Intelligence", "Prediction Lab", "Model Analytics"]
)

with overview_tab:
    st.subheader("City intelligence dashboard")
    st.caption("Understand how the selected city behaves over time and where its pollutant load stands in the selected year.")

    chart_col, profile_col = st.columns((1.35, 1))
    with chart_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        trend_fig = build_trend_chart(summary_df, selected_city, compare_city)
        st.altair_chart(trend_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with profile_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        profile_fig, profile_df = build_pollutant_profile_chart(selected_row, year_frame)
        st.altair_chart(profile_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    insight_col, leaderboard_col = st.columns((1, 1))
    with insight_col:
        st.markdown("### Review-ready talking points")
        top_profile = profile_df.iloc[0]
        render_insight(
            f"<strong>{selected_city}</strong> recorded <strong>{selected_row['AQI']:.1f}</strong> AQI in {selected_year}, "
            f"which is classified as <strong>{get_aqi_category(selected_row['AQI'])}</strong>."
        )
        render_insight(
            f"The city stands <strong>{abs(national_gap):.1f} AQI {'above' if national_gap > 0 else 'below'}</strong> "
            f"the same-year national city average of <strong>{national_avg:.1f}</strong>."
        )
        render_insight(
            f"The strongest pollutant signal is <strong>{top_profile['Pollutant']}</strong>, currently running at "
            f"<strong>{top_profile['Relative Level']:.2f}x</strong> the same-year national city average."
        )
        if compare_gap is not None:
            render_insight(
                f"Against <strong>{compare_city}</strong>, the selected city is "
                f"<strong>{abs(compare_gap):.1f} AQI {'higher' if compare_gap > 0 else 'lower'}</strong> in {selected_year}."
            )

    with leaderboard_col:
        top_polluted = year_frame[["City", "AQI"]].head(5).copy()
        top_polluted["AQI"] = top_polluted["AQI"].round(1)
        cleanest = year_frame[["City", "AQI"]].tail(5).sort_values("AQI").copy()
        cleanest["AQI"] = cleanest["AQI"].round(1)
        st.markdown("### Same-year leaderboard")
        left_table, right_table = st.columns(2)
        with left_table:
            st.markdown("**Most polluted cities**")
            st.dataframe(top_polluted, hide_index=True, width="stretch")
        with right_table:
            st.markdown("**Cleanest cities**")
            st.dataframe(cleanest, hide_index=True, width="stretch")

    st.markdown("### 2025 practical watchlist")
    watchlist_df = build_future_watchlist(summary_df)
    if not watchlist_df.empty:
        watchlist_display = watchlist_df.head(10).copy()
        watchlist_display["AQI 2025"] = watchlist_display["AQI 2025"].round(1)
        watchlist_display["Change vs 2024"] = watchlist_display["Change vs 2024"].round(1)
        st.dataframe(watchlist_display, hide_index=True, width="stretch")
        st.caption("This watchlist helps prioritize cities where intervention is most urgent for upcoming planning cycles.")
    else:
        st.info("Year 2025 data is not available yet.")

with prediction_tab:
    st.subheader("Interactive prediction lab")
    st.caption("Loaded with the selected city-year average profile. Adjust any pollutant value to simulate conditions instantly.")

    reference_profile = get_reference_profile(summary_df, selected_city, selected_year)
    input_columns = st.columns(4)
    input_values = {}

    for idx, feature in enumerate(FEATURE_COLUMNS):
        col = input_columns[idx % 4]
        default_value = float(reference_profile.get(feature, 0.0))
        input_values[feature] = col.number_input(
            feature,
            min_value=0.0,
            value=round(default_value, 2),
            step=0.1,
            key=f"{feature.replace('.', '').replace(' ', '_')}_{selected_city}_{selected_year}",
        )

    prediction = predict_aqi(model, input_values)
    load_df = describe_pollutant_load(input_values, reference_profile)
    dominant_pollutant = load_df.iloc[0]

    scenario_candidates = load_df["Pollutant"].head(3).tolist()
    scenario_rows = []
    for pollutant in scenario_candidates:
        scenario_result = simulate_reduction(model, input_values, pollutant, scenario_reduction)
        scenario_rows.append(
            {
                "Scenario": f"Reduce {pollutant} by {scenario_reduction}%",
                "Predicted AQI": scenario_result["aqi"],
                "Improvement": prediction["aqi"] - scenario_result["aqi"],
                "Category": scenario_result["category"],
            }
        )
    scenario_df = pd.DataFrame(scenario_rows).sort_values("Improvement", ascending=False)
    best_scenario = scenario_df.iloc[0]

    left_pred_col, right_pred_col = st.columns((1.05, 1))
    with left_pred_col:
        render_prediction_card(prediction)

    with right_pred_col:
        metric_cols = st.columns(2)
        metric_cols[0].metric("Prediction band", f"{prediction['interval_low']:.1f} - {prediction['interval_high']:.1f}")
        metric_cols[1].metric("Confidence", prediction["confidence"])
        metric_cols[0].metric("Dominant pollutant", dominant_pollutant["Pollutant"], f"{dominant_pollutant['Relative Load']:.2f}x baseline")
        metric_cols[1].metric("Best intervention", best_scenario["Scenario"], f"-{best_scenario['Improvement']:.1f} AQI")

        render_insight(
            f"The current input profile is most stressed on <strong>{dominant_pollutant['Pollutant']}</strong>, "
            f"running at <strong>{dominant_pollutant['Relative Load']:.2f}x</strong> the selected city-year baseline."
        )
        render_insight(
            f"If <strong>{best_scenario['Scenario']}</strong>, the predicted AQI drops to "
            f"<strong>{best_scenario['Predicted AQI']:.1f}</strong> and moves into the "
            f"<strong>{best_scenario['Category']}</strong> band."
        )
        render_insight(
            f"Health advisory: <strong>{get_health_advisory(prediction['category'])}</strong> "
            f"{get_response_tip(prediction['aqi'])}"
        )

    chart_col, scenario_col = st.columns((1.2, 1))
    with chart_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        levels_fig = build_level_chart(load_df)
        st.altair_chart(levels_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with scenario_col:
        st.markdown("### Mitigation simulator")
        scenario_display = scenario_df.copy()
        scenario_display["Predicted AQI"] = scenario_display["Predicted AQI"].round(1)
        scenario_display["Improvement"] = scenario_display["Improvement"].round(1)
        st.dataframe(
            scenario_display,
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The simulator reduces one pollutant at a time so you can explain which intervention offers the highest AQI improvement."
        )

    st.markdown("### Practical daily exposure planner")
    planner_col_1, planner_col_2, planner_col_3, planner_col_4 = st.columns(4)
    planned_activity = planner_col_1.selectbox(
        "Planned activity",
        ["Daily commute", "Walking", "Cycling", "Outdoor sports", "Outdoor work"],
    )
    planned_hours = planner_col_2.slider("Outdoor duration (hours)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    sensitive_group = planner_col_3.selectbox("Sensitive person involved?", ["No", "Yes"]) == "Yes"
    mask_used = planner_col_4.selectbox("Mask planned?", ["Yes", "No"]) == "Yes"

    exposure = evaluate_exposure_plan(
        aqi=prediction["aqi"],
        activity=planned_activity,
        outdoor_hours=planned_hours,
        sensitive_group=sensitive_group,
        mask_used=mask_used,
    )

    st.markdown(
        f"""
        <div class="prediction-card" style="background: #fffdfa; border-left: 8px solid {exposure['color']};">
            <div class="stat-label" style="color:#45544d;">Exposure planning result</div>
            <div style="font-size:1.6rem; font-weight:700; color:{exposure['color']};">{exposure['label']}</div>
            <div style="margin-top:0.5rem;">{exposure['message']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"- {exposure['steps'][0]}")
    st.markdown(f"- {exposure['steps'][1]}")

    st.markdown("### Intervention priority matrix")
    importance_df = benchmark["feature_importance"].copy()
    priority_df = load_df.merge(importance_df, left_on="Pollutant", right_on="Pollutant", how="left")
    max_importance = priority_df["Importance"].max() if not priority_df["Importance"].empty else 1.0
    priority_df["Priority Score"] = priority_df["Relative Load"] * (priority_df["Importance"] / max(max_importance, 1e-9))
    priority_display = priority_df[["Pollutant", "Relative Load", "Importance", "Priority Score"]].copy()
    priority_display["Relative Load"] = priority_display["Relative Load"].round(2)
    priority_display["Importance"] = priority_display["Importance"].round(3)
    priority_display["Priority Score"] = priority_display["Priority Score"].round(3)
    priority_display = priority_display.sort_values("Priority Score", ascending=False)
    st.dataframe(priority_display, hide_index=True, width="stretch")
    st.caption("Higher priority score means the pollutant is both elevated and strongly influential in AQI prediction.")

with analytics_tab:
    st.subheader("Model and dataset analytics")
    st.caption("These panels help explain the dataset, model quality, and pollutant relationships during the final review.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(df):,}")
    metric_cols[1].metric("Cities", f"{summary_df['City'].nunique()}")
    metric_cols[2].metric("Benchmark RMSE", f"{benchmark['rmse']:.2f}")
    metric_cols[3].metric("Benchmark R²", f"{benchmark['r2']:.3f}")

    model_col, corr_col = st.columns(2)
    with model_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        importance_fig = build_importance_chart(benchmark["feature_importance"])
        st.altair_chart(importance_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with corr_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        corr_fig = build_correlation_chart(df)
        st.altair_chart(corr_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    dist_col, notes_col = st.columns((1.1, 0.9))
    with dist_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        dist_fig = build_category_distribution(summary_df)
        st.altair_chart(dist_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with notes_col:
        st.markdown("### Evaluation summary")
        render_insight(
            f"The benchmark Random Forest reaches <strong>MAE {benchmark['mae']:.2f}</strong>, "
            f"<strong>RMSE {benchmark['rmse']:.2f}</strong>, and <strong>R² {benchmark['r2']:.3f}</strong> on a train-test split."
        )
        strongest_feature = benchmark["feature_importance"].iloc[0]
        render_insight(
            f"<strong>{strongest_feature['Pollutant']}</strong> is the strongest predictor in the benchmark model, "
            f"contributing <strong>{strongest_feature['Importance']:.3f}</strong> of total feature importance."
        )
        render_insight(
            f"The dataset tracks <strong>{summary_df['City'].nunique()}</strong> Indian cities from "
            f"<strong>{min(years)}</strong> to <strong>{max(years)}</strong>, giving the project both forecasting and comparative analysis value."
        )

st.caption(
    f"Latest dataset year: {latest_year}. Selected city profile: {selected_city} ({selected_year}). "
    "Use the sidebar to switch cities and demonstrate scenario analysis during your review."
)
