import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

ENV = os.getenv('ENV', "development")
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model_pipeline.joblib"))
FEATURES_PATH = os.getenv("FEAUTURES_PATH", str(BASE_DIR / "models" / "feature_schema.json" ))
HF_TOKEN = os.getenv("HF_TOKEN", "")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")