"""CASEFILE - Probable Location & Movement Analysis Dashboard."""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED, SYNTHETIC, MODELS
from src.prediction import load_location_model, predict_areas
from src.route_prediction import predict_route
from src.priority_scoring import rank_areas
from src.explainability import feature_importance, explanation_text
from src.mapping import make_map


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CASEFILE",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# LIGHTWEIGHT THEME-SAFE CSS
# =========================================================
# IMPORTANT:
# No forced background colors.
# No forced text colors.
# Streamlit controls Light/Dark mode.

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .case-subtitle {
        opacity: 0.65;
        font-size: 1rem;
        margin-top: -12px;
        margin-bottom: 18px;
    }

    .section-space {
        margin-top: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.title("🔎 CASEFILE")

st.markdown(
    '<div class="case-subtitle">'
    'Probable Location & Movement Analysis System'
    '</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Academic simulation only. Cases are fictional and model outputs "
    "are probabilistic. Anomaly flags represent statistical deviations "
    "and do not imply wrongdoing. This system must not be used to make "
    "real-world missing-person decisions."
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def artifacts():

    cases = pd.read_csv(
        SYNTHETIC / "fictional_cases.csv",
        parse_dates=["Last_Seen_Time"],
    )

    centers = pd.read_csv(
        PROCESSED / "area_centers.csv"
    )

    movement = pd.read_csv(
        PROCESSED / "movement_features.csv",
        parse_dates=["timestamp"],
    )

    transitions = json.loads(
        (MODELS / "transitions.json").read_text()
    )

    return cases, centers, movement, transitions


try:

    cases, centers, movement, transitions = artifacts()

    model = load_location_model()

except Exception as exc:

    st.error(
        "Artifacts are not ready. Run `py -m src.pipeline` first, "
        "then reload the dashboard."
    )

    with st.expander("Technical details"):
        st.exception(exc)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔎 CASEFILE")

    st.caption("Investigation Dashboard")

    st.divider()

    case_id = st.selectbox(
        "Select synthetic case",
        cases["Case_ID"],
    )

    case = cases.loc[
        cases["Case_ID"] == case_id
    ].iloc[0]

    st.divider()

    st.subheader("Dataset")

    st.metric(
        "GPS observations",
        f"{len(movement):,}"
    )

    st.metric(
        "Trajectories",
        f"{movement['trajectory_id'].nunique():,}"
    )

    st.metric(
        "Areas",
        f"{movement['area_id'].nunique():,}"
    )

    st.divider()

    st.caption(
        "CASEFILE v1.0 • Academic Simulation"
    )


# =========================================================
# RUN MODEL
# =========================================================

# =========================================================
# RUN MODEL
# =========================================================

predictions = predict_areas(
    model,
    case,
)

route = predict_route(
    transitions,
    case.Previous_Area,
)

current_hour = pd.to_datetime(case.Last_Seen_Time).hour

priorities = rank_areas(
    predictions,
    centers,
    case.Last_Latitude,
    case.Last_Longitude,
    route,
    anomalies=movement,
    current_hour=current_hour,
)

# =========================================================
# DATASET STATISTICS
# =========================================================

total_observations = len(movement)

anomalies = int(
    (movement["anomaly_label"] == -1).sum()
)

normal = total_observations - anomalies

anomaly_percentage = (
    anomalies / total_observations * 100
    if total_observations
    else 0
)

trajectory_count = movement["trajectory_id"].nunique()

area_count = movement["area_id"].nunique()


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

st.subheader("Dashboard Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="GPS Observations",
        value=f"{total_observations:,}",
    )

with k2:
    st.metric(
        label="Trajectories",
        value=f"{trajectory_count:,}",
    )

with k3:
    st.metric(
        label="Anomalies",
        value=f"{anomalies:,}",
    )

with k4:
    st.metric(
        label="Anomaly Rate",
        value=f"{anomaly_percentage:.1f}%",
    )


# =========================================================
# CASE INFORMATION
# =========================================================

st.divider()

st.subheader("📁 Case Information")

case_left, case_right = st.columns(
    [1, 2],
    gap="large",
)


with case_left:

    with st.container(border=True):

        st.markdown(f"### {case.Case_ID}")

        st.write(
            f"**Last known latitude:** "
            f"{case.Last_Latitude:.5f}"
        )

        st.write(
            f"**Last known longitude:** "
            f"{case.Last_Longitude:.5f}"
        )

        st.write(
            f"**Previous area:** "
            f"{case.Previous_Area}"
        )

        st.write(
            f"**Last seen:** "
            f"{case.Last_Seen_Time}"
        )


with case_right:

    with st.container(border=True):

        st.markdown("### Case Record")

        st.dataframe(
            case.to_frame("Value"),
            use_container_width=True,
            height=270,
        )


# =========================================================
# ANALYSIS TABS
# =========================================================

st.divider()

tab_location, tab_route, tab_anomaly, tab_explain = st.tabs(
    [
        "📍 Location Analysis",
        "🧭 Route Prediction",
        "🚨 Anomaly Analysis",
        "🧠 Explainability",
    ]
)


# =========================================================
# LOCATION ANALYSIS
# =========================================================

with tab_location:

    st.subheader("📍 Probable Areas")

    location_left, location_right = st.columns(
        2,
        gap="large",
    )

    with location_left:

        with st.container(border=True):

            st.markdown("### Model Probabilities")

            st.dataframe(
                predictions.head(5),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Top-k results are model probability estimates, "
                "not confirmed locations."
            )


    with location_right:

        with st.container(border=True):

            st.markdown("### Search Priority")

            priority_columns = [
                "area_id",
                "probability",
                "priority_score",
                "priority_label",
            ]

            available_columns = [
                column
                for column in priority_columns
                if column in priorities.columns
            ]

            st.dataframe(
                priorities[available_columns],
                use_container_width=True,
                hide_index=True,
            )


    st.subheader("🗺️ Interactive Movement Map")

    try:

        from streamlit_folium import st_folium

        fmap = make_map(
            case.Last_Latitude,
            case.Last_Longitude,
            centers,
            priorities,
            route,
        )

        st_folium(
            fmap,
            width=None,
            height=600,
        )

    except ImportError:

        st.warning(
            "The interactive map requires streamlit-folium."
        )


# =========================================================
# ROUTE PREDICTION
# =========================================================

with tab_route:

    st.subheader("🧭 Probable Movement Route")

    if route:

        route_df = pd.DataFrame(route)

        st.dataframe(
            route_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "The route is generated from observed "
            "area-to-area transition probabilities."
        )

    else:

        st.warning(
            "No observed transition is available from this area."
        )


# =========================================================
# ANOMALY ANALYSIS
# =========================================================

with tab_anomaly:

    st.subheader("🚨 Movement Anomaly Analysis")

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Normal Observations",
            f"{normal:,}",
        )

    with a2:

        st.metric(
            "Anomalous Observations",
            f"{anomalies:,}",
        )

    with a3:

        st.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.2f}%",
        )


    st.markdown("### Anomaly Distribution")

    chart_data = pd.DataFrame(
        {
            "Classification": [
                "Normal",
                "Anomaly",
            ],
            "Observations": [
                normal,
                anomalies,
            ],
        }
    )

    st.bar_chart(
        chart_data.set_index("Classification")
    )


    st.markdown("### Highest Anomaly Scores")

    anomaly_table = movement[
        movement["anomaly_label"] == -1
    ].copy()

    anomaly_table = anomaly_table.sort_values(
        "anomaly_score",
        ascending=False,
    )

    display_columns = [
        "person_id",
        "trajectory_id",
        "latitude",
        "longitude",
        "timestamp",
        "speed_kmh",
        "distance_km",
        "anomaly_score",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in anomaly_table.columns
    ]

    st.dataframe(
        anomaly_table[display_columns].head(25),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Isolation Forest identifies statistical deviations "
        "in movement features. An anomaly does not establish "
        "wrongdoing or a confirmed event."
    )


# =========================================================
# EXPLAINABILITY
# =========================================================

with tab_explain:

    st.subheader("🧠 Model Explainability")

    importance = feature_importance(model)

    explain_left, explain_right = st.columns(
        2,
        gap="large",
    )

    with explain_left:

        with st.container(border=True):

            st.markdown("### Feature Importance")

            st.bar_chart(
                importance
                .set_index("feature")
                .head(12)
            )


    with explain_right:

        with st.container(border=True):

            st.markdown("### Model Interpretation")

            st.info(
                explanation_text(
                    predictions,
                    importance,
                )
            )


    st.markdown("### Model Features")

    st.write(
        "The prediction system uses available case and movement "
        "information to estimate probable areas."
    )


# =========================================================
# EXPORT
# =========================================================

st.divider()

st.subheader("📥 Export Analysis")

export_left, export_right = st.columns(
    [1, 3],
)

with export_left:

    st.download_button(
        label="Download Priority CSV",
        data=priorities.to_csv(index=False),
        file_name="simulated_priority_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_right:

    st.caption(
        "Export contains the simulated area-priority ranking "
        "generated by the current model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CASEFILE • Probable Location & Movement Analysis • "
    "Academic Simulation"
)