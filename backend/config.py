import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

ENV = os.getenv('ENV', "development")
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model_bundle.pkl"))
FEATURES_PATH = os.getenv("FEAUTURES_PATH", str(BASE_DIR / "models" / "feature_schema.json" ))