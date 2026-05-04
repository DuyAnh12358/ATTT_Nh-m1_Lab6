import joblib

def load_model():
    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler