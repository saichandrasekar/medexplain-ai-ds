# medexplain-ai-ds
Datascience components for the medexplain-ai 

# MedExplain-AI DS — Model REST API

Trained on Heart Disease (Cleveland) dataset in Databricks.
Deployed via Render.

## Endpoints
- GET  /health   — health check
- POST /predict  — returns severity prediction

## Sample Request
curl -X POST "https://medexplain-ds.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "chestpain": 1,
    "chol": 233, "fbs": 1, "restecg": 2,
    "maxhr": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 3.0, "ca": 0.0, "thal": 2
  }'

-------

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

uvicorn backend.main:app --reload


curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":45,"sex":0,"cp":1,"trestbps":120,"chol":180,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":0.5,"slope":1,"ca":0,"thal":2}'


curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":65,"sex":1,"cp":3,"trestbps":160,"chol":300,"fbs":1,"restecg":2,"thalach":110,"exang":1,"oldpeak":3.5,"slope":3,"ca":3,"thal":3}'