import streamlit as st
from predict import load_model, predict

# Load the model once when the app starts
model = load_model()

st.title("Iris Flower Prediction")
st.write("Enter the features of the Iris flower to predict its species.")

# Input fields for features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
pedal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
pedal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Make prediction
if st.button("Predict"):
    features = [sepal_length, sepal_width, pedal_length, pedal_width]
    prediction = predict(model, features)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The predicted species is: **{species[prediction]}**")

# Footer with Model Metadata
import json
import os

meta_path = os.path.join(os.path.dirname(__file__), "model_meta.json")
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    st.markdown("---")
    st.markdown(
        f"**Version:** {meta['version']} • "
        f"**Best model:** {meta['best_model']} • "
        f"**MLflow run:** `{meta['mlflow_run_id']}` • "
        f"**Accuracy:** `{meta['metrics']['accuracy']:.3f}`"
    )
    st.markdown(f"[View in MLflow UI](http://localhost:5000)")
else:
    st.warning("Model metadata not found.")
