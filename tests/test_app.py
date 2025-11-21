import sys
import os
import pytest

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.predict import load_model, predict

def test_predict():
    """
    Test the predict function to ensure it returns a valid class id.
    """
    model = load_model()
    # Using a sample from the Iris dataset
    # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
    sample_features = [5.1, 3.5, 1.4, 0.2]
    
    prediction = predict(model, sample_features)
    
    # The Iris dataset has 3 classes: 0, 1, 2
    import numbers
    assert isinstance(prediction, numbers.Integral)
    assert prediction in [0, 1, 2]
