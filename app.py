import streamlit as st
import numpy as np
from joblib import load

# Load Model Files
model = load('model.joblib')
scaler = load('scaler.joblib')
encoder = load('encoder.joblib')

# Page Settings
st.set_page_config(
page_title="Carbon Emission Predictor",
page_icon="🌍"
)

# Title

st.title("🌍 Carbon Emission Prediction")
st.subheader("Sustainability Project")

st.write("Predict Carbon Emission Level using Machine Learning")

st.divider()

# Input Fields

country = st.selectbox("Country", encoder.classes_)

year = st.slider("Year", 1990, 2030, 2020)

population = st.number_input(
"Population (Millions)",
1.0,
2000.0,
500.0
)

gdp = st.number_input(
"GDP (Billion USD)",
100.0,
50000.0,
10000.0
)

energy = st.number_input(
"Energy Consumption (TWh)",
100.0,
50000.0,
10000.0
)

renewable = st.slider(
"Renewable Energy %",
0.0,
100.0,
30.0
)

st.divider()

# Prediction Button

if st.button("Predict Emission"):

    country_enc = encoder.transform([country])[0]

    data = np.array([[
        country_enc,
        year,
        population,
        gdp,
        energy,
        renewable
    ]])

    data = scaler.transform(data)

    pred = model.predict(data)[0]

    if pred == 1:
        st.error("🔴 High Carbon Emission Expected")
    else:
        st.success("🟢 Low Carbon Emission Expected")

st.divider()

st.caption("Green Skilling & Sustainability ML Project")