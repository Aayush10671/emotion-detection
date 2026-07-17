import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Get token from environment
dagshub_token = os.getenv("DAGSHUB_PAT")
mlflow_enabled = False

if dagshub_token:
    try:
        # Set MLflow tracking URI with authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        repo_owner = "Aayush10671"
        repo_name = "emotion-detection"
        
        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')
        
        # Test connection
        mlflow.search_experiments()
        mlflow_enabled = True
        logger.info("MLflow connection successful!")
    except Exception as e:
        logger.warning(f"MLflow connection failed: {e}")
        logger.info("Continuing without MLflow logging...")
        mlflow_enabled = False
else:
    logger.warning("DAGSHUB_PAT not found. MLflow logging disabled.")

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def log_to_mlflow(metrics: dict, clf, X_test: np.ndarray):
    """Log metrics and model to MLflow if enabled."""
    if not mlflow_enabled:
        logger.info("MLflow logging skipped (not enabled)")
        return
    
    try:
        # Only set experiment if MLflow is enabled
        mlflow.set_experiment("dvc-pipeline")
        
        with mlflow.start_run() as run:
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model
            mlflow.sklearn.log_model(clf, "model")
            
            # Log test data info
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("features", X_test.shape[1])
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')
            
            logger.info("Metrics logged to MLflow successfully")
            logger.info(f"Run ID: {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")

def main():
    try:
        logger.info("Starting model evaluation...")
        
        # Load model and data
        clf = load_model('./models/model.pkl')
        
        # FIXED: Using test_bow.csv instead of test_tfidf.csv
        test_data = load_data('./data/features/test_bow.csv')
        
        logger.info(f"Test data shape: {test_data.shape}")
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)
        logger.info(f"Metrics: {metrics}")
        
        # Save metrics locally
        save_metrics(metrics, 'reports/metrics.json')
        
        # Try to log to MLflow (optional, won't fail the pipeline)
        log_to_mlflow(metrics, clf, X_test)
        
        logger.info("Model evaluation completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Make sure the file exists at ./data/features/test_bow.csv")
        raise
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == '__main__':
    main()