import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,0.85); }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #102a43 0%, #174a70 58%, #256b91 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2.1rem; }
    .hero p { color: #dbeafe; margin: 0; font-size: 1rem; }
    .result-card {
        padding: 1.35rem 1.5rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e4eaf2;
        box-shadow: 0 5px 18px rgba(16,42,67,.06);
    }
    .result-label { color: #526477; font-size: .9rem; margin-bottom: .25rem; }
    .result-price { color: #123b5d; font-size: 2.2rem; font-weight: 750; }
    div[data-testid="stForm"] { background: white; padding: 1.2rem; border-radius: 16px; border: 1px solid #e4eaf2; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).resolve().parent / "flight_price_model.pkl"


@st.cache_resource(show_spinner="Loading your trained model...")
def load_artifacts(path: str):
    artifacts = joblib.load(path)
    required = {"model", "OHE", "scaler"}
    missing = required.difference(artifacts.keys())
    if missing:
        raise KeyError(f"The PKL file is missing these objects: {', '.join(sorted(missing))}")
    return artifacts["model"], artifacts["OHE"], artifacts["scaler"]


st.markdown(
    """
    <div class="hero">
        <h1>✈️ Flight Price Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place **flight_price_model.pkl** in the same folder as **app.py**."
    )
    st.stop()

try:
    model, ohe, scaler = load_artifacts(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Unable to load the model file: {exc}")
    st.stop()

# Dropdown options come directly from the fitted OneHotEncoder categories.
# This keeps the app aligned with the categories the model learned during training.
category_options = {
    column: [str(value) for value in values]
    for column, values in zip(ohe.feature_names_in_, ohe.categories_)
}

# Use the ranges learned by the fitted MinMaxScaler for numerical inputs.
num_columns = list(scaler.feature_names_in_)
num_min = dict(zip(num_columns, scaler.data_min_))
num_max = dict(zip(num_columns, scaler.data_max_))

st.sidebar.title("About this app")
st.sidebar.write(
    "Flight Price Predictor estimates the ticket fare for a journey based on the "
    "flight details you provide. Select the airline, route, travel class, number "
    "of stops, departure and arrival times, flight duration, and days remaining "
    "before departure to receive an estimated price in Indian rupees."
)
st.sidebar.write(
    "Use the estimate to plan your travel budget, compare possible flight options, "
    "and get an idea of how your selected journey details may relate to the ticket fare."
)
st.sidebar.info(
    "The result is an estimated fare, not a live airline quotation. Actual prices "
    "can change depending on availability, demand, booking time, and airline updates."
)

st.subheader("Enter flight details")
st.caption("Choose the flight information below, then select **Predict flight price**.")

with st.form("flight_details_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("Airline", category_options["airline"])
        flight = st.selectbox("Flight number", category_options["flight"])
        source_city = st.selectbox("Source city", category_options["source_city"])

    with col2:
        destination_options = [
            city for city in category_options["destination_city"] if city != source_city
        ]
        destination_city = st.selectbox("Destination city", destination_options)
        departure_time = st.selectbox("Departure time", category_options["departure_time"])
        arrival_time = st.selectbox("Arrival time", category_options["arrival_time"])

    with col3:
        stops = st.selectbox(
            "Number of stops",
            category_options["stops"],
            format_func=lambda value: {
                "zero": "Non-stop",
                "one": "1 stop",
                "two_or_more": "2 or more stops",
            }.get(value, value),
        )
        travel_class = st.selectbox("Travel class", category_options["class"])

        duration_min = float(num_min.get("duration", 0.83))
        duration_max = float(num_max.get("duration", 49.83))
        duration = st.number_input(
            "Duration (hours)",
            min_value=duration_min,
            max_value=duration_max,
            value=min(max(2.5, duration_min), duration_max),
            step=0.1,
            help="Enter the total flight duration in hours.",
        )

    days_min = int(num_min.get("days_left", 1))
    days_max = int(num_max.get("days_left", 49))
    days_left = st.slider(
        "Days left until departure",
        min_value=days_min,
        max_value=days_max,
        value=min(max(15, days_min), days_max),
        help="How many days before departure the ticket is being considered.",
    )

    submitted = st.form_submit_button(
        "✈️ Predict flight price", type="primary", use_container_width=True
    )

if submitted:
    if source_city == destination_city:
        st.warning("Source city and destination city must be different.")
        st.stop()

    # Keep the raw column names and order identical to the encoder's fit data.
    categorical_input = pd.DataFrame(
        [[
            airline,
            flight,
            source_city,
            departure_time,
            stops,
            arrival_time,
            destination_city,
            travel_class,
        ]],
        columns=list(ohe.feature_names_in_),
    )

    numerical_input = pd.DataFrame(
        [[duration, days_left]],
        columns=list(scaler.feature_names_in_),
    )

    try:
        categorical_encoded = ohe.transform(categorical_input)
        numerical_scaled = scaler.transform(numerical_input)

        # Join the encoded categorical and scaled numerical features.
        model_input = pd.concat([categorical_encoded, numerical_scaled], axis=1)

        # Match the exact feature order used when the model was trained.
        model_input = model_input.reindex(columns=model.feature_names_in_, fill_value=0)

        predicted_price = float(model.predict(model_input)[0])

        st.markdown("### Your estimated fare")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">PREDICTED FLIGHT PRICE</div>
                <div class="result-price">₹ {predicted_price:,.0f}</div>
                <div style="color:#64748b; font-size:.88rem; margin-top:.35rem;">
                    Estimated from the details you entered. Actual airline fares may vary.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Review your selected flight details"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "Flight detail": [
                            "Airline", "Flight", "Source", "Destination",
                            "Departure", "Arrival", "Stops", "Class",
                            "Duration (hours)", "Days left",
                        ],
                        "Selected value": [
                            airline, flight, source_city, destination_city,
                            departure_time, arrival_time, stops, travel_class,
                            duration, days_left,
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.info(
            "Check that the PKL file was created from the same preprocessing workflow "
            "and that its model, encoder, and scaler were saved together."
        )

st.divider()
st.caption(
    "Disclaimer: This is an educational machine learning project. Predictions are estimates, not live airline quotations."
)
