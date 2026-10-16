
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Household Electricity Classifier",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Household Electricity Classifier")

st.write(
    "Explore household electricity measurements and "
    "classify an observation as High or Low consumption."
)

st.info(
    "This is a classification demonstration using supplied "
    "measurements. It does not forecast future electricity use."
)

# --------------------------------------------------
# 2. Load the saved model and scaler
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"
SCALER_PATH = BASE_DIR / "models" / "standard_scaler.joblib"

@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error(
        "Model files not found. Please save the trained model "
        "and scaler inside the models folder."
    )
    st.stop()

model, scaler = load_model_files()


# --------------------------------------------------
# Dataset Insights and Visualizations
# --------------------------------------------------

st.divider()
st.header("📊 Explore Electricity Consumption")

st.write(
    "Explore the household electricity dataset through "
    "the visualizations generated during data analysis."
)

assets_dir = BASE_DIR / "assets"

# Display daily consumption chart
st.subheader("1. Daily Electricity Consumption")
daily_chart = assets_dir / "daily_consumption.png"

if daily_chart.exists():
    st.image(str(daily_chart), use_container_width=True)
else:
    st.warning("Daily consumption chart not found.")

# Display hourly consumption chart
st.subheader("2. Consumption by Hour of Day")
hourly_chart = assets_dir / "hourly_consumption.png"

if hourly_chart.exists():
    st.image(str(hourly_chart), use_container_width=True)
else:
    st.warning("Hourly consumption chart not found.")

# Display class distribution
st.subheader("3. High vs Low Consumption")
class_chart = assets_dir / "class_distribution.png"

if class_chart.exists():
    st.image(str(class_chart), use_container_width=True)
else:
    st.warning("Class distribution chart not found.")

# Display model comparison
st.subheader("4. Classification Model Comparison")
comparison_chart = assets_dir / "model_comparison.png"

if comparison_chart.exists():
    st.image(str(comparison_chart), use_container_width=True)
else:
    st.warning("Model comparison chart not found.")

st.divider()


# Keep the same six features and order as model training
FEATURES = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

# --------------------------------------------------
# 3. Collect user input
# --------------------------------------------------

st.subheader("Enter Electricity Measurements")

with st.form("electricity_form"):

    col1, col2 = st.columns(2)

    with col1:
        reactive_power = st.number_input(
            "Global reactive power (kW)",
            min_value=0.0,
            max_value=1.39,
            value=0.1,
            step=0.01
        )

        voltage = st.number_input(
            "Voltage (V)",
            min_value=200.0,
            max_value=270.0,
            value=240.0,
            step=0.1
        )

        intensity = st.number_input(
            "Global intensity (A)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.1
        )

    with col2:
        sub1 = st.number_input(
            "Kitchen sub-metering (Wh)",
            min_value=0.0,
            max_value=90.0,
            value=0.0,
            step=1.0
        )

        sub2 = st.number_input(
            "Laundry sub-metering (Wh)",
            min_value=0.0,
            max_value=85.0,
            value=1.0,
            step=1.0
        )

        sub3 = st.number_input(
            "Water heating / AC sub-metering (Wh)",
            min_value=0.0,
            max_value=35.0,
            value=6.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Classify Consumption",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 4. Predict the consumption class
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame(
        [[
            reactive_power,
            voltage,
            intensity,
            sub1,
            sub2,
            sub3
        ]],
        columns=FEATURES
    )

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Classification Result")

    if prediction == "High":
        st.warning("Predicted class: HIGH consumption")
    else:
        st.success("Predicted class: LOW consumption")

    st.caption(
        "This result depends on the trained model and the "
        "measurements entered. It is not a direct measurement "
        "of actual consumption."
    )

# --------------------------------------------------
# 5. Project information
# --------------------------------------------------

with st.expander("About this project"):
    st.write(
        "Dataset: UCI Individual Household Electric Power "
        "Consumption."
    )

    st.write(
        "Model: Logistic Regression, trained on six "
        "electricity-related features."
    )

    st.write(
        "The High and Low labels were created using the median "
        "Global Active Power in the project dataset."
    )

st.divider()

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
