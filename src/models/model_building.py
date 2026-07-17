import numpy as np
import pandas as pd
import os
import logging
import pickle
import json
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from dotenv import load_dotenv
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")

mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

def main():
    try:
        logger.info("Starting model training...")
        
        # params = yaml.safe_load(open('params.yaml','r'))['model_building']
        
        train_data = pd.read_csv('./data/features/train_bow.csv')
        logger.info(f"Train data shape: {train_data.shape}")
        
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        
        # clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        
        logger.info("Fitting model...")
        clf.fit(X_train, y_train)
        
        os.makedirs('models', exist_ok=True)
        pickle.dump(clf, open('models/model.pkl', 'wb'))
        
        logger.info("Model saved successfully to models/model.pkl")
        
        # Log model to MLflow
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(clf, "model", registered_model_name="sentiment_classifier")
            logger.info(f"Model logged to MLflow with run_id: {run.info.run_id}")
            
            # Save experiment info for later registration
            experiment_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }
            os.makedirs('reports', exist_ok=True)
            with open('reports/experiment_info.json', 'w') as f:
                json.dump(experiment_info, f)
            logger.info("Experiment info saved to reports/experiment_info.json")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()