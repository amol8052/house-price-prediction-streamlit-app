import os
import joblib
import pandas as pd
import streamlit as st

# --------------------
# App config (MUST be the first Streamlit call)
# --------------------
st.set_page_config(page_title="House Price Demo", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction — Demo")

# --------------------
# Utilities
# --------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None

def make_input_df(values: dict):
    return pd.DataFrame([values])

# --------------------
# Model load
# --------------------
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models/house_rf.joblib"))
model = load_model(MODEL_PATH)

# --------------------
# Sidebar: controls & presets
# --------------------
st.sidebar.header("Input Controls")

preset = st.sidebar.selectbox("Choose a preset example", ("Default", "Low-income area", "High-income area", "Urban dense"))
if preset == "Default":
    defaults = {"MedInc": 5.0, "HouseAge": 20.0, "AveRooms": 5.0, "AveBedrms": 1.0, "Population": 1000.0, "AveOccup": 2.5, "Latitude": 34.0, "Longitude": -118.0}
elif preset == "Low-income area":
    defaults = {"MedInc": 2.5, "HouseAge": 40.0, "AveRooms": 4.0, "AveBedrms": 1.2, "Population": 3000.0, "AveOccup": 3.5, "Latitude": 35.0, "Longitude": -117.0}
elif preset == "High-income area":
    defaults = {"MedInc": 10.0, "HouseAge": 10.0, "AveRooms": 7.0, "AveBedrms": 1.0, "Population": 500.0, "AveOccup": 2.0, "Latitude": 37.0, "Longitude": -122.0}
else:  # Urban dense
    defaults = {"MedInc": 6.0, "HouseAge": 25.0, "AveRooms": 3.0, "AveBedrms": 1.1, "Population": 10000.0, "AveOccup": 4.0, "Latitude": 28.6, "Longitude": 77.2}

st.sidebar.markdown("### Feature sliders (adjust values)")
MedInc = st.sidebar.slider("MedInc (median income)", min_value=0.0, max_value=20.0, value=float(defaults["MedInc"]), step=0.1)
HouseAge = st.sidebar.slider("HouseAge (years)", min_value=0.0, max_value=100.0, value=float(defaults["HouseAge"]), step=1.0)
AveRooms = st.sidebar.slider("AveRooms", min_value=1.0, max_value=20.0, value=float(defaults["AveRooms"]), step=0.1)
AveBedrms = st.sidebar.slider("AveBedrms", min_value=0.0, max_value=10.0, value=float(defaults["AveBedrms"]), step=0.1)
Population = st.sidebar.number_input("Population", min_value=0.0, value=float(defaults["Population"]), step=1.0, format="%.0f")
AveOccup = st.sidebar.slider("AveOccup", min_value=0.0, max_value=20.0, value=float(defaults["AveOccup"]), step=0.1)
Latitude = st.sidebar.number_input("Latitude", value=float(defaults["Latitude"]), step=0.01, format="%.2f")
Longitude = st.sidebar.number_input("Longitude", value=float(defaults["Longitude"]), step=0.01, format="%.2f")

# Extra: show raw inputs toggle
show_inputs = st.sidebar.checkbox("Show raw input dataframe", value=False)

# --------------------
# Main layout: two columns
# --------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input preview")
    input_values = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    if show_inputs:
        st.dataframe(make_input_df(input_values))
    st.markdown("Click **Predict** to run the model on the values above.")

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded — cannot predict.")
        else:
            # basic validation
            if Population < 0:
                st.error("Population must be >= 0")
            else:
                x = make_input_df(input_values)
                try:
                    pred = model.predict(x)[0]
                    st.success(f"Predicted median house value: **{pred:.3f}**")
                    st.info("Model: RandomForestRegressor (sklearn). This is a demo prediction for learning purposes.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Explain & Tips")
    st.write("""
    - **MedInc**: median income in tens of thousands (higher → higher house value).
    - **HouseAge**: average age of houses in the block.
    - **AveRooms / AveBedrms**: average rooms/bedrooms per household.
    - **Population / AveOccup**: population and average occupants.
    - **Latitude / Longitude**: location information.
    """)
    st.markdown("---")
    st.write("Try different presets from the sidebar to see how predictions change.")
    st.write("You can dockerize this app or deploy the API separately for production.")

st.markdown("---")
st.caption("Local demo — not production. Model trained on California housing sample data.")
