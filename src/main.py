from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Gauge
from model_prophet import get_test_data, train
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import pickle
import time


app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Histogramme pour mesurer le temps de réponse de /predict
PREDICT_RESPONSE_TIME = Histogram(
    "predict_response_time_seconds",
    "Temps de réponse de l'endpoint /predict",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# Gauges pour stocker les métriques de performance du modèle
R2_SCORE = Gauge("model_r2_score", "Coefficient de détermination du modèle")
RMSE = Gauge("model_rmse", "Root Mean Squared Error du modèle")
MAPE = Gauge("model_mape", "Mean Absolute Percentage Error du modèle")

train()

# Charger le modèle Prophet
with open("./models/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)
    
# Charger les données de test pour évaluer le modèle
df_test = get_test_data()
df_test.rename(columns={"Date": "ds", "Consommation": "y"}, inplace=True)

@app.get("/")
def main():
    return "hello worldss"

@app.get("/predict/")
def predict(days: int = 30):
    start_time = time.time()
    future = model.make_future_dataframe(periods=days, freq='D')
    forecast = model.predict(future)
    duration = time.time() - start_time
    
    PREDICT_RESPONSE_TIME.observe(duration)
    
    return forecast[['ds', 'yhat']].tail(days).to_dict(orient="records")

@app.get("/metrics/model")
def compute_model_metrics():
    """ Calcule les métriques de performance du modèle """
    future = model.predict(df_test[['ds']])
    y_true = df_test['y'].values
    y_pred = future['yhat'].values

    # Calcul des métriques
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Mettre à jour les métriques Prometheus
    R2_SCORE.set(r2)
    RMSE.set(rmse)
    MAPE.set(mape)
    
    return {"r2_score": r2, "rmse": rmse, "mape": mape}