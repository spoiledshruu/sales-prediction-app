import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_model.pkl")

# Streamlit UI
st.title("📊 Sales Prediction App")
st.markdown("Enter your advertising budget to predict product sales.")

# Input sliders
tv = st.slider("TV Advertising Budget", 0, 300, 150)
radio = st.slider("Radio Advertising Budget", 0, 50, 25)
news = st.slider("Newspaper Advertising Budget", 0, 100, 20)

# Predict button
if st.button("Predict Sales"):
    features = np.array([[tv, radio, news]])
    prediction = model.predict(features)[0]
    st.success(f"💡 Predicted Sales: **{prediction:.2f} units**")
