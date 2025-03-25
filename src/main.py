from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import pickle

app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Charger le modèle Prophet
with open("./models/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/predict/")
def predict(days: int = 30):
    future = model.make_future_dataframe(periods=days, freq='D')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(days).to_dict(orient="records")