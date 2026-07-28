# medexplain-ai-ds
Datascience components for the medexplain-ai 

# MedExplain-AI DS — Model REST API

Trained on Heart Disease (Cleveland) dataset in Databricks.
Deployed via Render.

## Endpoints
- GET  /health   — health check
- POST /predict  — returns severity prediction

## Sample Request
curl -X POST "https://your-render-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "chestpain": 1,
    "chol": 233, "fbs": 1, "restecg": 2,
    "maxhr": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 3.0, "ca": 0.0, "thal": 2
  }'
