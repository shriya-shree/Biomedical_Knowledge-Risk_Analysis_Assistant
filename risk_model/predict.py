import joblib
import numpy as np

model = joblib.load("diabetes_risk_model.pkl")

def risk_level(prob):
    if prob < 0.33:
        return "Low"
    elif prob < 0.66:
        return "Medium"
    else:
        return "High"

def predict_risk(input_features):
    prob = model.predict_proba([input_features])[0][1]
    return risk_level(prob), prob
