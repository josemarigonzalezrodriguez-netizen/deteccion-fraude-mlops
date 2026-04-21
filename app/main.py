from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(
    title="Detector de Fraude Bancario",
    description="Prediccion de transacciones fraudulentas con LightGBM + MLflow",
    version="1.0"
)

# Cargar el mejor modelo directamente por run_id de Optuna trial_0
MODEL_URI = "runs:/a9562c1bd61d48ae96cf75cc7bb97526/model"
model = mlflow.pyfunc.load_model(MODEL_URI)

class Transaccion(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

class Prediccion(BaseModel):
    prediccion: int
    etiqueta: str
    probabilidad: float

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}

@app.post("/predict", response_model=Prediccion)
def predict(transaccion: Transaccion):
    df = pd.DataFrame([transaccion.dict()])
    proba = float(model.predict(df)[0])
    pred  = int(proba > 0.5)
    return Prediccion(
        prediccion=pred,
        etiqueta="FRAUDE" if pred == 1 else "LEGÍTIMA",
        probabilidad=round(proba, 4)
    )