
import joblib
import sys

try:
    model = joblib.load("app/model.joblib")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
