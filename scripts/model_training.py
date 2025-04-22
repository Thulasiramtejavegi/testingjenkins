import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def train_model():
    print("Training model...")
    
    # Load data
    data_path = 'data/processed/churn_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Split into features and target
    X = data.drop('churn', axis=1)
    y = data['churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_training") as run:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Log model - this is the important part
        mlflow.sklearn.log_model(model, "model")
        
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model locally
        run_id = run.info.run_id
        model_path = f"models/model_{run_id}.pkl"
        mlflow.sklearn.save_model(model, model_path)
        
        print(f"Model saved to {model_path}")
        
        # Log local model file as an artifact
        mlflow.log_artifact(model_path)
        
        return run_id

if __name__ == "__main__":
    train_model()
