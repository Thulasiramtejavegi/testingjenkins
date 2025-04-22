import pandas as pd
import numpy as np
import os
import mlflow

def prepare_data():
    # Create a sample dataset for demonstration
    # In a real scenario, you would load your data from a database or file
    
    print("Preparing data...")
    
    # Create directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # For demo, create a synthetic churn dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    age = np.random.normal(45, 15, n_samples)
    tenure = np.random.gamma(2, 10, n_samples)
    monthly_charges = np.random.uniform(30, 120, n_samples)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_samples)
    
    # Generate target (churn)
    logit = -2 + 0.02 * age - 0.1 * tenure + 0.03 * monthly_charges
    probability = 1 / (1 + np.exp(-logit))
    churn = np.random.binomial(1, probability)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })
    
    # Save data
    data.to_csv('data/processed/churn_data.csv', index=False)
    print(f"Data prepared and saved. Shape: {data.shape}")
    
    return data

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Log parameters
    with mlflow.start_run(run_name="data_preparation"):
        data = prepare_data()
        mlflow.log_param("n_samples", len(data))
        mlflow.log_param("n_features", data.shape[1] - 1)
        mlflow.log_param("churn_rate", data['churn'].mean())
