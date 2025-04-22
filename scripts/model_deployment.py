import os
import mlflow
import mlflow.sklearn
import json
import time

def deploy_model():
    print("Deploying model...")
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get latest model version
    client = mlflow.tracking.MlflowClient()
    
    # First check if model exists
    try:
        client.get_registered_model("ChurnModel")
        print("Found registered model 'ChurnModel'")
    except Exception as e:
        print(f"Model ChurnModel does not exist yet: {e}")
        print("Creating new model...")
        client.create_registered_model("ChurnModel")
        
        # Get the latest run to register
        runs = client.search_runs(
            experiment_ids=["0"],
            filter_string="tags.mlflow.runName = 'model_training'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            run_id = runs[0].info.run_id
            # Create new version
            try:
                client.create_model_version(
                    name="ChurnModel",
                    source=f"runs:/{run_id}/model",
                    run_id=run_id
                )
                print("Created initial model version")
            except Exception as e:
                print(f"Error creating model version: {e}")
                # Try with local model path
                model_path = f"models/model_{run_id}.pkl"
                if os.path.exists(model_path):
                    try:
                        client.create_model_version(
                            name="ChurnModel",
                            source=model_path,
                            run_id=run_id
                        )
                        print("Created initial model version from local path")
                    except Exception as e2:
                        print(f"Error creating model version from local path: {e2}")
    
    # Get latest version with better error handling
    try:
        latest_versions = client.get_latest_versions("ChurnModel", stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            print(f"Found latest version: {latest_version}")
            
            # Transition model to production
            try:
                client.transition_model_version_stage(
                    name="ChurnModel",
                    version=latest_version,
                    stage="Production"
                )
                print(f"Model version {latest_version} deployed to production")
            except Exception as e:
                print(f"Error transitioning model: {e}")
        else:
            print("No model versions found")
            # Try to find any versions in any stage
            all_versions = client.get_latest_versions("ChurnModel")
            if all_versions:
                latest_version = all_versions[0].version
                print(f"Found version in another stage: {latest_version}")
            else:
                print("Creating a default version 1")
                latest_version = "1"
    except Exception as e:
        print(f"Error getting model versions: {e}")
        print("Using default version 1")
        latest_version = "1"
    
    # Create deployment config
    deployment_config = {
        "model_name": "ChurnModel",
        "model_version": latest_version,
        "deployment_timestamp": int(time.time() * 1000),  # Use time.time() instead of mlflow.utils.get_current_time_millis()
        "environment": "production"
    }
    
    # Create directory if it doesn't exist
    os.makedirs('models/deployment', exist_ok=True)
    
    # Save deployment config
    config_path = 'models/deployment/deployment_config.json'
    with open(config_path, 'w') as f:
        json.dump(deployment_config, f, indent=2)
    
    print(f"Deployment config saved to {config_path}")
    
    # Start MLflow run for deployment
    with mlflow.start_run(run_name="model_deployment"):
        mlflow.log_artifact(config_path)
        mlflow.log_param("model_version", latest_version)
        mlflow.log_param("deployment_environment", "production")
        print("Deployment run logged to MLflow")

if __name__ == "__main__":
    deploy_model()
