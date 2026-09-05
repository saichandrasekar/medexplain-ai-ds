import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import joblib
import pandas as pd
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, create_model
from backend.config import MODEL_PATH, FEATURES_PATH
# from backend.services.rag_service import _get_collection
# from backend.services.embedding_service import get_embeddings

router = APIRouter()

_model = None
_feature_list = None
_model_version = None
PredictRequest: type[BaseModel] = None

def _load_artifacts():
    global _model, _feature_list, _model_version, PredictRequest
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r") as f:
            features_meta = json.load(f)
        _feature_list = features_meta["feature_names"]
        _model_version = str(features_meta["model_version"])
        # Build Pydantic model after feature list is known
        fields = {name: (float, ...) for name in _feature_list}
        PredictRequest = create_model("PredictRequest", **fields)

def _scale_risk_score(raw_score: float) -> float:
    return round(max(0.0, min(1.0, raw_score)) * 10, 2)

def _risk_label(score: float) -> str:
    if score < 4:
        return "low"
    if score < 7:
        return "medium"
    return "high"

@router.post("/predict")
def predict(payload: Dict) -> Dict:
    try:
        input_df = pd.DataFrame(
            [[payload[name] for name in _feature_list]],
            columns=_feature_list
        )
        raw_score = float(_model.predict_proba(input_df)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    score = _scale_risk_score(raw_score)
    return {
        "risk_score": score,
        "risk_level": _risk_label(score),
        "model_version": _model_version
    }

@router.get("/model/info")
def model_info() -> Dict:
    
    return {
        "features": _feature_list,
        "model_version": _model_version
    }

# @router.get("/vector/test")
# def test_vector_store() -> Dict:
#     results = _get_collection().query(
#         query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
#         n_results=2 # how many results to return
#     )
#     print(results)

#     return {
#         "results_length": len(results.get('ids')[0])
#     }

# @router.get('/embed/test')
# def test_embedding() -> Dict:
#     texts = [
#         "This is a sample document",
#         "Another piece of text to embed",
#         "Query: What is machine learning?"
#     ]

#     embeddings = get_embeddings(texts, batch_size=32)
#     print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")

#     return {
#         "length": embeddings.shape
#     }