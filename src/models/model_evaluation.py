import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")

mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_data(file_path):
    return pd.read_csv(file_path)

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

def save_metrics(metrics, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)

def save_model_info(run_id, model_path, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump({'run_id': run_id, 'model_path': model_path}, file, indent=4)

def main():
    mlflow.set_experiment("dvc-pipeline")
    
    with mlflow.start_run() as run:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/features/test_bow.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        metrics = evaluate_model(clf, X_test, y_test)
        
        save_metrics(metrics, 'reports/metrics.json')
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.log_artifact('reports/metrics.json')
        
        print(f"Model evaluation completed. Run ID: {run.info.run_id}")

if __name__ == '__main__':
    main()