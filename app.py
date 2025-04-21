import streamlit as st
import pickle
import numpy as np

# Load model and preprocessing tools
model = pickle.load(open("iris_model.pkl", "rb"))
scaler = pickle.load(open("iris_scaler.pkl", "rb"))
encoder = pickle.load(open("iris_encoder.pkl", "rb"))

st.title("🌸 Iris Species Prediction App")

st.markdown("Enter the flower's measurements below to predict the species.")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    species = encoder.inverse_transform(pred)
    st.success(f"The predicted Iris species is: **{species[0]}**")
