# SUML: A Simple Machine Learning Project

This project is a simple machine learning application that demonstrates how to train a model and use it for predictions in a web application.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```
python 3.9
pip
```

### Installation

A step by step series of examples that tell you how to get a development env running:

1. Clone the repo:
   ```
   git clone https://github.com/nightfury394/SUML
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the application

To run the Streamlit application:

```
streamlit run app/app.py
```

## Project Structure

*   `app/app.py`: The main file for the Streamlit web application.
*   `app/predict.py`: Contains the function to make predictions using the trained model.
*   `app/model.joblib`: The trained machine learning model.
*   `train_model.py`: The script to train the machine learning model.
*   `requirements.txt`: The list of Python dependencies.
*   `.github/workflows/main.yml`: The GitHub Actions workflow for continuous integration.

## Technologies Used

*   [Python](https://www.python.org/)
*   [scikit-learn](https://scikit-learn.org/)
*   [Streamlit](https://streamlit.io/)
*   [Joblib](https://joblib.readthedocs.io/)
*   [GitHub Actions](https://github.com/features/actions)
