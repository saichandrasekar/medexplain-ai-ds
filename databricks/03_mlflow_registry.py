import mlflow
from mlflow.tracking import MlflowClient

# Set the experiment
experiment_name = "/Users/sairoot.2024@gmail.com/medexplain-health-risk"
mlflow.set_experiment(experiment_name)

# Get the experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Query experiment runs and sort by AUC-ROC (descending to get best first)
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.auc_roc DESC"],
    max_results=1
)

if len(runs) == 0:
    raise ValueError("No runs found in the experiment")

# Get the best run
best_run = runs.iloc[0]
best_run_id = best_run.run_id
best_auc_roc = best_run["metrics.test_auc_roc"]

print(f"Best run ID: {best_run_id}")
print(f"Best AUC-ROC: {best_auc_roc}")

# Get the model URI from the best run
model_uri = f"runs:/{best_run_id}/model"

# Register the model
# model_name = "risk-evaluator-model"
model_name = "workspace.medexplain-ai.risk-evaluator-model"
registered_model = mlflow.register_model(model_uri, model_name)

print(f"\nRegistered model '{model_name}' version {registered_model.version}")

# Initialize MLflow client
client = MlflowClient()

# Add description to the registered model version
model_description = (
    "Health risk prediction model for the MedExplain project. "
    "This model predicts patient health risk levels based on clinical features. "
    f"Trained on medical data with best AUC-ROC score of {best_auc_roc:.4f}."
)

client.update_model_version(
    name=model_name,
    version=registered_model.version,
    description=model_description
)

print(f"\nAdded description to model version {registered_model.version}")

client.set_registered_model_alias(
    name=model_name,
    alias="staging",
    version=registered_model.version
)
print(f"Alias 'staging' set on version {registered_model.version}")

# Add tags to the model version
client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="deployment_alias",
    value="staging"
)

client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="dataset",
    value="medexplain-health-data"
)

client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="version",
    value="v1.0"
)

client.set_model_version_tag(
    name=model_name,
    version=registered_model.version,
    key="author",
    value="sairoot.2024@gmail.com"
)

print(f"\nAdded tags to model version {registered_model.version}")
print("\nModel registration complete!")