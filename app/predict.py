import joblib
import numpy as np
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)

def predict(model, features):
    # Ensure features are in the correct format (e.g., numpy array)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]
