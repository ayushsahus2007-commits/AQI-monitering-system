import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

st.set_page_config(
    page_title="AQI Monitoring System v2",
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
                    radial-gradient(circle at top left, rgba(199, 109, 60, 0.10), transparent 26%),
                    radial-gradient(circle at top right, rgba(51, 92, 75, 0.14), transparent 24%),
                    linear-gradient(180deg, #fcf7ef 0%, #f7f1e6 100%);
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
            }
            .insight-card {
                background: #fffdfa;
                border-left: 5px solid var(--clay);
                border-radius: 14px;
                padding: 14px 16px;
                margin-bottom: 0.8rem;
                box-shadow: 0 8px 22px rgba(73, 77, 71, 0.06);
            }
            .prediction-card {
                border-radius: 20px;
                padding: 22px;
                color: #1f2522;
                box-shadow: 0 14px 35px rgba(73, 77, 71, 0.1);
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
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f6efe3 0%, #efe5d3 100%);
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


def format_delta(delta: float | None, lower_is_better: bool = True) -> str:
    if delta is None:
        return "No previous-year record"
    direction = "improved" if (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better) else "increased"
    return f"{delta:+.1f} AQI ({direction})"


def build_trend_chart(summary_df: pd.DataFrame, selected_city: str, compare_city: str | None):
    national_trend = summary_df.groupby("Year", as_index=False)["AQI"].mean()
    city_trend = summary_df[summary_df["City"] == selected_city].sort_values("Year")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(city_trend["Year"], city_trend["AQI"], marker="o", linewidth=3, color="#c76d3c", label=selected_city)
    ax.plot(
        national_trend["Year"],
        national_trend["AQI"],
        marker="o",
        linewidth=2.2,
        linestyle="--",
        color="#335c4b",
        label="National city average",
    )

    if compare_city:
        compare_trend = summary_df[summary_df["City"] == compare_city].sort_values("Year")
        ax.plot(
            compare_trend["Year"],
            compare_trend["AQI"],
            marker="o",
            linewidth=2.4,
            color="#8aa37b",
            label=compare_city,
        )

    ax.set_title("Historical AQI trajectory")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average AQI")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig


def build_pollutant_profile_chart(selected_row: pd.Series, year_frame: pd.DataFrame):
    national_profile = year_frame[FEATURE_COLUMNS].mean()
    profile_df = pd.DataFrame(
        {
            "Pollutant": FEATURE_COLUMNS,
            "Relative Level": [selected_row[feature] / max(national_profile[feature], 1.0) for feature in FEATURE_COLUMNS],
        }
    ).sort_values("Relative Level")

    colors = ["#c76d3c" if value > 1 else "#8aa37b" for value in profile_df["Relative Level"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.barh(profile_df["Pollutant"], profile_df["Relative Level"], color=colors)
    ax.axvline(1.0, color="#335c4b", linestyle="--", linewidth=1.5)
    ax.set_title("Pollutant fingerprint vs same-year national profile")
    ax.set_xlabel("Relative intensity")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig, profile_df.sort_values("Relative Level", ascending=False)


def build_level_chart(load_df: pd.DataFrame):
    chart_df = load_df.sort_values("Value")
    positions = range(len(chart_df))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.barh([pos - 0.18 for pos in positions], chart_df["Baseline"], height=0.34, color="#8aa37b", label="City-year baseline")
    ax.barh([pos + 0.18 for pos in positions], chart_df["Value"], height=0.34, color="#c76d3c", label="Current input")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(chart_df["Pollutant"])
    ax.set_title("Current pollutant inputs vs selected-city baseline")
    ax.set_xlabel("Concentration")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig


def build_importance_chart(importance_df: pd.DataFrame):
    chart_df = importance_df.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(chart_df["Pollutant"], chart_df["Importance"], color="#335c4b")
    ax.set_title("Model feature importance")
    ax.set_xlabel("Importance score")
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig


def build_correlation_chart(df: pd.DataFrame):
    corr_df = (
        df[FEATURE_COLUMNS + ["AQI"]]
        .corr(numeric_only=True)["AQI"]
        .drop("AQI")
        .sort_values()
        .reset_index()
        .rename(columns={"index": "Pollutant", "AQI": "Correlation"})
    )

    colors = ["#c76d3c" if value > 0 else "#8aa37b" for value in corr_df["Correlation"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(corr_df["Pollutant"], corr_df["Correlation"], color=colors)
    ax.set_title("Correlation of pollutants with AQI")
    ax.set_xlabel("Pearson correlation")
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig


def build_category_distribution(summary_df: pd.DataFrame):
    dist_df = summary_df.copy()
    dist_df["Category"] = dist_df["AQI"].apply(get_aqi_category)
    ordered_categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    counts = (
        dist_df["Category"]
        .value_counts()
        .reindex(ordered_categories, fill_value=0)
        .reset_index()
        .rename(columns={"index": "Category", "Category": "Count"})
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(counts["Category"], counts["Count"], color=["#2e8b57", "#87a330", "#d28b17", "#cc5a2f", "#9a3d3d", "#5a2a2a"])
    ax.set_title("Distribution of city-year AQI categories")
    ax.set_ylabel("Number of city-year records")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.patch.set_facecolor("#fffdfa")
    ax.set_facecolor("#fffdfa")
    return fig


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
        <div class="hero-kicker">Final Review Dashboard Upgrade</div>
        <div class="hero-title">AQI Monitoring System v2</div>
        <div class="hero-copy">
            A polished forecasting and analytics workspace for Indian city air quality data.
            This version adds benchmarking, prediction confidence, mitigation simulation, and model analytics
            so the project feels closer to a deployable decision-support system than a single-output demo.
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

with st.expander("What changed in version 2"):
    st.write(
        """
        - Multi-section dashboard with city intelligence, live prediction lab, and model analytics.
        - New peer benchmarking using national average, city ranks, and side-by-side trend comparison.
        - Prediction confidence band derived from the Random Forest ensemble spread.
        - Scenario simulator that estimates AQI improvement after pollutant reduction actions.
        - Cleaner code structure with reusable AQI utility functions and cached data/model loading.
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
        st.pyplot(trend_fig, use_container_width=True)
        plt.close(trend_fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with profile_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        profile_fig, profile_df = build_pollutant_profile_chart(selected_row, year_frame)
        st.pyplot(profile_fig, use_container_width=True)
        plt.close(profile_fig)
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
            st.dataframe(top_polluted, hide_index=True, use_container_width=True)
        with right_table:
            st.markdown("**Cleanest cities**")
            st.dataframe(cleanest, hide_index=True, use_container_width=True)

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
        st.pyplot(levels_fig, use_container_width=True)
        plt.close(levels_fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with scenario_col:
        st.markdown("### Mitigation simulator")
        scenario_display = scenario_df.copy()
        scenario_display["Predicted AQI"] = scenario_display["Predicted AQI"].round(1)
        scenario_display["Improvement"] = scenario_display["Improvement"].round(1)
        st.dataframe(
            scenario_display,
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "The simulator reduces one pollutant at a time so you can explain which intervention offers the highest AQI improvement."
        )

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
        st.pyplot(importance_fig, use_container_width=True)
        plt.close(importance_fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with corr_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        corr_fig = build_correlation_chart(df)
        st.pyplot(corr_fig, use_container_width=True)
        plt.close(corr_fig)
        st.markdown("</div>", unsafe_allow_html=True)

    dist_col, notes_col = st.columns((1.1, 0.9))
    with dist_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        dist_fig = build_category_distribution(summary_df)
        st.pyplot(dist_fig, use_container_width=True)
        plt.close(dist_fig)
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
