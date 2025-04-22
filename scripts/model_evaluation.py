import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def evaluate_model():
    print("Evaluating model...")
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get latest run ID
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.mlflow.runName = 'model_training'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No training runs found")
    
    run_id = runs[0].info.run_id
    print(f"Found run ID: {run_id}")
    
    # Try to load the model from the local saved file first
    model_path = f"models/model_{run_id}.pkl"
    if os.path.exists(model_path):
        print(f"Loading model from local path: {model_path}")
        model = mlflow.sklearn.load_model(model_path)
    else:
        print(f"Local model not found, attempting to load from MLflow")
        try:
            # Try with artifacts path
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            # As a fallback, try to load from the run directory
            artifacts_uri = runs[0].info.artifact_uri
            model_uri = os.path.join(artifacts_uri, "model")
            print(f"Trying to load from: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
    
    # Load data
    data_path = 'data/processed/churn_data.csv'
    data = pd.read_csv(data_path)
    
    # Split into features and target
    X = data.drop('churn', axis=1)
    y = data['churn']
    
    # Get prediction probabilities
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Cross-validation for more robust evaluation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Create directory if it doesn't exist
    os.makedirs('models/evaluation', exist_ok=True)
    
    # Start MLflow run for evaluation
    with mlflow.start_run(run_name="model_evaluation") as run:
        # Log standard metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log cross-validation metrics
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std", cv_scores.std())
        
        # Log model parameters explicitly
        mlflow.log_param("model_type", type(model).__name__)
        if hasattr(model, 'n_estimators'):
            mlflow.log_param("n_estimators", model.n_estimators)
        
        # Create and log visualization artifacts
        
        # 1. ROC Curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_path = 'models/evaluation/roc_curve.png'
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = 'models/evaluation/confusion_matrix.png'
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        # 3. Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = X.columns
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            fi_path = 'models/evaluation/feature_importance.png'
            plt.savefig(fi_path)
            mlflow.log_artifact(fi_path)
            plt.close()
        
        # 4. Classification Report as text file
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = 'models/evaluation/classification_report.csv'
        report_df.to_csv(report_path)
        mlflow.log_artifact(report_path)
        
        print(f"Model evaluated with ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Also log the actual model as an artifact in this run for traceability
        eval_model_path = f"models/evaluation/evaluated_model_{run.info.run_id}.pkl"
        joblib.dump(model, eval_model_path)
        mlflow.log_artifact(eval_model_path)
        
        # Create model registry entry if it doesn't exist
        try:
            # Check if model exists
            client.get_registered_model("ChurnModel")
            print("Found existing registered model 'ChurnModel'")
        except Exception:
            # Create model if it doesn't exist
            client.create_registered_model("ChurnModel")
            print("Created registered model 'ChurnModel'")
        
        # Log model version for deployment
        try:
            model_version = client.create_model_version(
                name="ChurnModel",
                source=f"runs:/{run_id}/model",
                run_id=run_id
            )
            print(f"Model registered as version: {model_version.version}")
        except Exception as e:
            print(f"Error registering model version: {e}")
            print("Trying with local model path")
            try:
                model_version = client.create_model_version(
                    name="ChurnModel",
                    source=model_path,
                    run_id=run_id
                )
                print(f"Model registered as version: {model_version.version}")
            except Exception as e:
                print(f"Error registering model with local path: {e}")
        
        return run.info.run_id

if __name__ == "__main__":
    evaluate_model()
