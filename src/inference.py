import joblib
import pandas as pd
from .preprocess import FEATURES

def predict(input_data, model_path="models/best_model.joblib"):
    model = joblib.load(model_path)
    frame = pd.DataFrame(input_data, columns=FEATURES)
    return model.predict(frame)
