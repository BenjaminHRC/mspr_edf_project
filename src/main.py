from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prophet.diagnostics import performance_metrics, cross_validation
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

@app.get("/metrics/prophet")
def metrics_prophet():
    df_cv = cross_validation(model, initial='2190 days', period='30 days', horizon = "365 days")
    return performance_metrics(df_cv).to_dict(orient="records")