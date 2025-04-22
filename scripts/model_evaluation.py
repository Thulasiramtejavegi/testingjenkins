import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_model():
    print("Evaluating model...")
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
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
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Start MLflow run for evaluation
    with mlflow.start_run(run_name="model_evaluation"):
        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Create directory if it doesn't exist
        os.makedirs('models/evaluation', exist_ok=True)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save plot
        roc_path = 'models/evaluation/roc_curve.png'
        plt.savefig(roc_path)
        
        # Log artifact
        mlflow.log_artifact(roc_path)
        
        print(f"Model evaluated with ROC AUC: {roc_auc:.4f}")
        
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

if __name__ == "__main__":
    evaluate_model()
