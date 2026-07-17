import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging
import requests

# Configure logging properly
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger('data_ingestion')  # Fixed typo: 'data_digestion' -> 'data_ingestion'

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        test_size = params['data_ingestion']['test_size']
        logger.info('Test size retrieved successfully: %s', test_size)
        return test_size
    except FileNotFoundError:
        logger.error('params.yaml not found, using default test_size=0.2')
        return 0.2
    except KeyError:
        logger.error('data_ingestion.test_size not found in params.yaml, using default=0.2')
        return 0.2
    except Exception as e:
        logger.error('Unexpected error loading params: %s', str(e))
        return 0.2

def read_data(url: str) -> pd.DataFrame:
    try:
        logger.info('Reading data from URL: %s', url)
        # Add timeout to prevent hanging indefinitely
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        logger.info('Successfully loaded data from URL, shape: %s', df.shape)
        return df
    except requests.Timeout:
        logger.error('Timeout: URL request took too long (>30 seconds)')
        return pd.DataFrame()
    except Exception as e:
        logger.error('Error reading data from %s: %s', url, str(e))
        return pd.DataFrame()

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if 'tweet_id' in df.columns:
            df.drop(columns=['tweet_id'], inplace=True)
        else:
            logger.warning('tweet_id column not found, skipping drop')
            
        if 'sentiment' not in df.columns:
            logger.error('sentiment column not found in dataframe')
            return pd.DataFrame()
            
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'] = final_df['sentiment'].map({'happiness': 1, 'sadness': 0})
        
        logger.info('Data processed successfully, final shape: %s', final_df.shape)
        return final_df
    except Exception as e:
        logger.error('Error processing data: %s', str(e))
        return pd.DataFrame()

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info('Data saved successfully to %s', data_path)
        logger.debug('Train shape: %s, Test shape: %s', train_data.shape, test_data.shape)
    except Exception as e:
        logger.error('Error saving data: %s', str(e))

def main():
    logger.info('Starting data ingestion pipeline')
    
    test_size = load_params('params.yaml')
    logger.info('Using test_size: %s', test_size)
    
    df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    
    if df.empty:
        logger.error('DataFrame is empty, exiting pipeline')
        return
        
    final_df = process_data(df)
    
    if final_df.empty:
        logger.error('Processed DataFrame is empty, exiting pipeline')
        return
        
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    logger.info('Train-test split completed - Train: %s, Test: %s', train_data.shape, test_data.shape)
    
    data_path = os.path.join("data", "raw")
    save_data(data_path, train_data, test_data)
    
    logger.info('Data ingestion pipeline completed successfully')

if __name__ == "__main__":
    main()