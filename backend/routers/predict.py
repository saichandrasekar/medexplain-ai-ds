import json
import pickle
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, create_model

from backend.config import MODEL_PATH, FEATURES_PATH

import pandas as pd

router = APIRouter()

try:
    with open(MODEL_PATH, "rb") as f:
        model_bundle = pickle.load(f)
    
    with open(FEATURES_PATH, 'r') as f:
        features_meta = json.load(f)

    feature_list = features_meta['feature_names']
    # model_version = features_meta.get('model_version', getattr(model_bundle, "version"))
    model_version = features_meta['model_version']

except FileNotFoundError as exc:
     raise RuntimeError(
          f"Failed to load model artifacts: {exc}."
          f"Check MODEL_PATH/FEATURES_PATH in config.py"
     )
fields = {name: (float, ...) for name in feature_list}
PredictRequest: type[BaseModel] = create_model("PredictRequest", **fields)
    
def _scale_risk_score(raw_score: float) -> float:
    """Scale raw model output (assumed 0-1) to 0-10"""
    return round(max(0.0, min(1.0, raw_score)) * 10, 2)

def _risk_label(score_0_10: float) -> str:
    if score_0_10 < 4:
        return "low"

    if score_0_10 < 7:
        return "medium"

    return "high"

@router.post("/predict")
def predict(payload: PredictRequest) -> Dict:
    ordered_input = [[getattr(payload, name) for name in feature_list]]

    print(type(model_bundle))
    print(dir(model_bundle))



    try:
        input_df = pd.DataFrame(ordered_input, columns=feature_list)
        raw_model = model_bundle.get_raw_model()
        raw_score = float(raw_model.predict_proba(input_df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    score = _scale_risk_score(raw_score)

    return{
        "risk_score": float(score),
        "risk_level": _risk_label(score),
        "model_version": str(model_version)
    }

@router.get("/model/info")
def model_info() -> Dict:
    return {
        "features": feature_list,
        "model_version": model_version
    }