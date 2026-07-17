import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting feature engineering...")
        
        with open('params.yaml', 'r') as f:
            max_features = yaml.safe_load(f)['feature_engineering']['max_features']
        
        train_data = pd.read_csv('./data/processed/train_processed_data.csv')
        test_data = pd.read_csv('./data/processed/test_processed_data.csv')
        
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        data_path = os.path.join("data", "features")
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
        
        logger.info(f"Done! Train shape: {X_train_bow.shape}, Test shape: {X_test_bow.shape}")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main()