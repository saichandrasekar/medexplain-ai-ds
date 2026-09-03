import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Start MLflow experiment
mlflow.set_experiment("/Users/sairoot.2024@gmail.com/medexplain-health-risk")

# COMMAND ----------

# Load CSV file
df = spark.read.csv(
    "/Volumes/workspace/medexplain-ai/medexplain/processed/heart_disease_processed.csv",
    header=True,
    inferSchema=True
)

# Display the first few rows
display(df)

# COMMAND ----------

# Start an MLflow run using context manager
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("param_name", "param_value")
    
    # Log metrics
    mlflow.log_metric("metric_name", 0.95)
    
    # Your training/evaluation code here
    # ...
    
    print("MLflow run completed")

# COMMAND ----------



# Convert Spark DataFrame to pandas for scikit-learn/XGBoost
df_pandas = df.toPandas()

# Define features and target
# Adjust 'target' to your actual target column name
target_column = 'target'
X = df_pandas.drop(columns=[target_column])
y = df_pandas[target_column]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

from category_encoders import OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(use_cat_names=True))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data (80/20 train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create pipeline with preprocessing and XGBoost
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Train model with MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    
    # Create signature and log model
    signature_input = X_train.head(100)
    signature = infer_signature(signature_input, model.predict(signature_input))
    
    # Calculate AUC-ROC (handle binary and multiclass)
    if len(np.unique(y_test)) == 2:
        # Binary classification
        test_auc_roc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
    else:
        # Multiclass classification
        test_auc_roc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
    
    # Log all metrics
    mlflow.log_metric("test_auc_roc", test_auc_roc)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC-ROC: {test_auc_roc:.4f}")
    
    # Log feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get feature importance from the XGBoost classifier in the pipeline
    xgb_model = model.named_steps['classifier']
    feature_importance = xgb_model.feature_importances_
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    
    # Get feature names from the preprocessor
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    except:
        feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
    
    ax.barh(range(len(indices)), feature_importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Feature Importances')
    ax.invert_yaxis()
    plt.tight_layout()
    
    # Save and log feature importance plot
    feature_importance_path = "/tmp/feature_importance.png"
    plt.savefig(feature_importance_path)
    mlflow.log_artifact(feature_importance_path, "plots")
    plt.close()
    
    # Log confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    # Save and log confusion matrix
    confusion_matrix_path = "/tmp/confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path, "plots")
    plt.close()
    
    # Log the model with signature and input example
    signature_input = X_test.head(100)
    signature = infer_signature(signature_input, model.predict(signature_input))
    
    model_info = mlflow.sklearn.log_model(
        model,
        name="risk-evaluator-model",
        signature=signature,
        input_example=X_test.head(3)
    )
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Artifacts logged: feature_importance.png, confusion_matrix.png")
    print(f"\nModel logged at: {model_info.model_uri}")