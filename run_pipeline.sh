#!/bin/bash

# Start MLflow server
echo "Starting MLflow server..."
docker-compose up -d mlflow
sleep 5

# Run data preparation
echo "Running data preparation..."
docker-compose run training python -m scripts.data_preparation

# Run model training
echo "Running model training..."
docker-compose run training python -m scripts.model_training

# Run model evaluation
echo "Running model evaluation..."
docker-compose run training python -m scripts.model_evaluation

# Run model deployment
echo "Running model deployment..."
docker-compose run training python -m scripts.model_deployment

echo "Pipeline completed successfully!"
