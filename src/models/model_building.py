import numpy as np
import pandas as pd
import os
import logging
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting model training...")
        
        params = yaml.safe_load(open('params.yaml','r'))['model_building']
        
        train_data = pd.read_csv('./data/features/train_tfidf.csv')
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
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()