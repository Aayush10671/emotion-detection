import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting model evaluation...")
        
        clf = pickle.load(open('models/model.pkl', 'rb'))
        test_data = pd.read_csv('./data/features/test_tfidf.csv')
        logger.info(f"Test data shape: {test_data.shape}")
        
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        
        logger.info("Making predictions...")
        y_pred = clf.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Metrics: {metrics}")
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/metrics.json', 'w') as file:
            json.dump(metrics, file, indent=4)
        
        logger.info("Metrics saved to reports/metrics.json")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()