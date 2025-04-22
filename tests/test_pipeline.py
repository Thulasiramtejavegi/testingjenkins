import pytest
import os
import pandas as pd
import numpy as np

def test_data_preparation():
    # Import here to avoid importing when tests are collected
    from scripts.data_preparation import prepare_data
    
    # Run data preparation
    data = prepare_data()
    
    # Check if data file exists
    assert os.path.exists('data/processed/churn_data.csv')
    
    # Check data properties
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert 'churn' in data.columns
    
    # Check data types
    assert data['age'].dtype == np.float64
    assert data['churn'].isin([0, 1]).all()

def test_model_training():
    # Create mock data if it doesn't exist
    if not os.path.exists('data/processed/churn_data.csv'):
        from scripts.data_preparation import prepare_data
        prepare_data()
    
    # Import here to avoid importing when tests are collected
    from scripts.model_training import train_model
    
    # Run model training
    run_id = train_model()
    
    # Check if model file exists
    assert os.path.exists(f'models/model_{run_id}.pkl')
