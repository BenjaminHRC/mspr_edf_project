from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

# Charger le modèle Prophet
with open("models/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict/")
def predict(days: int = 30):
    future = model.make_future_dataframe(periods=days, freq='D')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(days).to_dict(orient="records")