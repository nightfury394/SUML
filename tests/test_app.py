from app.predict import predict, load_model
import numpy as np

def test_predict():
    model = load_model()
    # Example features for Iris dataset (sepal length, sepal width, petal length, petal width)
    features = [5.1, 3.5, 1.4, 0.2]
    prediction = predict(model, features)
    assert isinstance(prediction, (int, np.integer))
    assert prediction in [0, 1, 2]