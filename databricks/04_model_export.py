%pip install category_encoders xgboost

import mlflow
import mlflow.pyfunc
import pickle
import json
import pandas as pd

# Step 1: Load model from registry using URI
model_name = "workspace.medexplain-ai.risk-evaluator-model"
model_version = 1
model_uri = f"models:/{model_name}/{model_version}"

print(f"Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print("✓ Model loaded successfully")

# Step 2: Run a smoke test - predict on 3 sample inputs
print("\n--- Running Smoke Test ---")

sample_data = pd.DataFrame([
    # Normal patient
    {
        "age": 45.0, "sex": 0.0, "cp": 1.0, "trestbps": 120.0,
        "chol": 180.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 0.5, "slope": 1.0, "ca": 0.0, "thal": 2.0
    },
    # Moderate risk
    {
        "age": 55.0, "sex": 1.0, "cp": 2.0, "trestbps": 140.0,
        "chol": 240.0, "fbs": 1.0, "restecg": 1.0, "thalach": 130.0,
        "exang": 1.0, "oldpeak": 1.5, "slope": 2.0, "ca": 1.0, "thal": 3.0
    },
    # High risk
    {
        "age": 65.0, "sex": 1.0, "cp": 3.0, "trestbps": 160.0,
        "chol": 300.0, "fbs": 1.0, "restecg": 2.0, "thalach": 110.0,
        "exang": 1.0, "oldpeak": 3.5, "slope": 3.0, "ca": 3.0, "thal": 3.0
    }
])

print(f"Sample input shape: {sample_data.shape}")
predictions = model.predict(sample_data)
print(f"Predictions: {predictions}")
print(f"Output shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
print("✓ Smoke test passed")

# Step 3: Save as pickle to DBFS Volumes
export_path = "/Volumes/workspace/medexplain-ai/medexplain/export/model_bundle.pkl"
print(f"\n--- Saving model to {export_path} ---")
with open(export_path, 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as pickle")

# Step 4: Save feature schema as JSON
feature_schema_path = "/Volumes/workspace/medexplain-ai/medexplain/export/feature_schema.json"
print(f"\n--- Saving feature schema to {feature_schema_path} ---")

# Extract feature names from model metadata
try:
    model_info = mlflow.models.get_model_info(model_uri)
    if model_info.signature and model_info.signature.inputs:
        feature_names = [input.name for input in model_info.signature.inputs.inputs]
    else:
        feature_names = list(sample_data.columns)  # Fallback to sample data columns
except:
    feature_names = list(sample_data.columns)  # Fallback

feature_schema = {
    "model_name": model_name,
    "model_version": model_version,
    "feature_names": feature_names,
    "num_features": len(feature_names)
}

with open(feature_schema_path, 'w') as f:
    json.dump(feature_schema, f, indent=2)
print("✓ Feature schema saved")
print(f"Feature schema: {json.dumps(feature_schema, indent=2)}")

# Step 5: Download instructions
print("\n" + "="*60)
print("📥 DOWNLOAD INSTRUCTIONS")
print("="*60)
print(f"Model pickle file: {export_path}")
print(f"Feature schema file: {feature_schema_path}")
print("\nTo download these files:")
print("1. Use Databricks CLI:")
print(f"   databricks fs cp dbfs:{export_path} ./model_bundle.pkl")
print(f"   databricks fs cp dbfs:{feature_schema_path} ./feature_schema.json")
print("\n2. Or download via Workspace UI:")
print("   - Navigate to Catalog > workspace > medexplain-ai > medexplain > export")
print("   - Click on the files to download")
print("="*60)