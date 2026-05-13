# Detección de fraude bancario (MLOps + MLflow + FastAPI)

Proyecto para el challenge de MLOps orientado a **experimentación con MLflow** (logging de parámetros/métricas/modelos) y **puesta en producción** del modelo con **FastAPI**.

## Dataset

Se utiliza el dataset **Credit Card Fraud Detection** (Kaggle: `mlg-ulb/creditcardfraud`), con cientos de miles de transacciones y un problema fuertemente desbalanceado.

El proyecto contempla dos formas de obtener los datos:

- **Descarga desde Kaggle** vía `kagglehub` (`src/load_data.py`).
- **DVC** (el dataset está referenciado como `data/raw/creditcard.csv.dvc`).

## Estructura del repo

- `src/load_data.py`: descarga y guarda `data/raw/creditcard.csv`.
- `src/train.py`: entrenamiento + búsqueda de hiperparámetros con Optuna y logging en MLflow.
- `app/main.py`: API FastAPI para predicción usando un modelo registrado como artefacto en MLflow.
- `dvc.yaml`: pipeline DVC (`load_data` y `train`).
- `mlflow.db`: backend store local (SQLite) para el tracking.

## Requisitos

- Python 3.10+ (recomendado)
- Dependencias Python: `pandas`, `scikit-learn`, `lightgbm`, `mlflow`, `optuna`, `fastapi`, `uvicorn`, `kagglehub`
- (Opcional) `dvc` si quieres reproducir el pipeline con DVC

Instalación rápida:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install pandas scikit-learn lightgbm mlflow optuna fastapi uvicorn kagglehub
```

## MLflow (tracking server)

El entrenamiento (`src/train.py`) usa como tracking URI `http://localhost:5000`, por lo que primero hay que levantar el servidor:

```bash
mlflow server ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root ./mlartifacts ^
  --host 127.0.0.1 ^
  --port 5000
```

Notas:
- Si es la primera vez, se creará la carpeta `mlartifacts/` al ejecutar el servidor/experimentos (está en `.gitignore`).
- Alternativamente puedes cambiar el tracking URI en `src/train.py` si usas otro servidor.

## 1) Cargar datos

### Opción A: Descarga con `kagglehub` (recomendada)

```bash
python src/load_data.py
```

Esto genera `data/raw/creditcard.csv`.

### Opción B: DVC

Si tienes configurado un remoto válido en tu entorno:

```bash
dvc pull
```

## 2) Entrenamiento y experimentación (Optuna + MLflow)

Lanza una optimización con Optuna (por defecto `n_trials=15`) y loggea cada trial en MLflow:

```bash
python src/train.py
```

Qué se loggea en cada ejecución (run):
- Parámetros del modelo (`mlflow.log_params`)
- Métricas: `auc_roc`, `f1_score`, `precision`, `recall`
- Modelo LightGBM como artefacto MLflow (`mlflow.lightgbm.log_model`)

Para explorar resultados:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## 3) Puesta en producción (FastAPI)

La API carga un modelo desde MLflow usando `MODEL_URI` en `app/main.py`.

Importante:
- Ahora mismo `MODEL_URI` está fijado a un `run_id` concreto (`runs:/.../model`).
- Si vuelves a entrenar en otra máquina/entorno, tendrás que **actualizar ese `run_id`** por el del mejor experimento en tu MLflow.

Arranque del servicio:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predicción (ejemplo con payload completo del dataset):

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"Time\":0,\"V1\":0,\"V2\":0,\"V3\":0,\"V4\":0,\"V5\":0,\"V6\":0,\"V7\":0,\"V8\":0,\"V9\":0,\"V10\":0,\"V11\":0,\"V12\":0,\"V13\":0,\"V14\":0,\"V15\":0,\"V16\":0,\"V17\":0,\"V18\":0,\"V19\":0,\"V20\":0,\"V21\":0,\"V22\":0,\"V23\":0,\"V24\":0,\"V25\":0,\"V26\":0,\"V27\":0,\"V28\":0,\"Amount\":0}"
```

Respuesta:
- `prediccion`: `0/1`
- `etiqueta`: `"LEGÍTIMA"` o `"FRAUDE"`
- `probabilidad`: probabilidad estimada

## Reproducibilidad con DVC (opcional)

El pipeline está definido en `dvc.yaml`:

```bash
dvc repro
```

Si en tu entorno el comando `python3` no existe (típico en Windows), ajusta `dvc.yaml` para usar `python` o ejecuta los scripts manualmente con `python ...`.

## Entregables

Para el challenge se entrega:
- Este repositorio (código + configuración).
- Un PDF explicando: dataset elegido, proceso de experimentación, cómo se usó MLflow para guiar decisiones, métricas comparadas, modelo final seleccionado y cómo se desplegó con FastAPI (incluyendo pruebas con `curl`).
