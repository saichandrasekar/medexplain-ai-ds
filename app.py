from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="MedExplain Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model bundle at startup
with open("model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
FEATURES  = bundle["features"]

SEVERITY_MAP = {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect"}


class PatientInput(BaseModel):
    age:      float
    sex:      int
    chestpain: int
    restbp:   float = 0.0  # was empty in dataset, default to 0
    chol:     float
    fbs:      int
    restecg:  int
    maxhr:    float
    exang:    int
    oldpeak:  float
    slope:    float
    ca:       float
    thal:     int


@app.get("/")
def root():
    return {
        "service": "MedExplain Predict API",
        "status":  "running",
        "features": FEATURES
    }


@app.get("/health")
def health():
    return {"status": "ok","version": "1.0.0", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        X = np.array([[
            patient.age, patient.sex, patient.chestpain,
            patient.restbp, patient.chol, patient.fbs,
            patient.restecg, patient.maxhr, patient.exang,
            patient.oldpeak, patient.slope, patient.ca,
            patient.thal
        ]])

        encoded     = pipeline.predict(X)[0]
        proba       = pipeline.predict_proba(X)[0]
        severity    = SEVERITY_MAP.get(int(encoded), "Unknown")

        return {
            "prediction":  int(encoded),
            "severity":    severity,
            "confidence":  round(float(np.max(proba)), 3),
            "class_probabilities": {
                SEVERITY_MAP[i]: round(float(p), 3)
                for i, p in enumerate(proba)
            },
            "source": "databricks-mlflow-sklearn"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
