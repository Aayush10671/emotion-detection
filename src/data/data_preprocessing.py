import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer , WordNetLemmatizer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger('data_preprocessing')


# Download NLTK data only once with error handling
def setup_nltk():
    """Download required NLTK data with timeout and error handling."""
    try:
        logger.info('Checking NLTK data...')
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info('NLTK data ready')
    except Exception as e:
        logger.warning(f'NLTK download warning: {e}. Proceeding anyway...')

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.content.iloc[i].split()) < 3:
            df.content.iloc[i] = np.nan

def normalize_text(df):
    """Normalize the text data."""
    try:
        logger.info('Starting text normalization for %d rows', len(df))
        # Use 'content' instead of 'text' since that's the actual column name
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.info('Text normalization completed successfully')
        return df
    except Exception as e:
        logger.error(f"Error in normalize_text: {e}")
        raise

if __name__ == "__main__":
    # Setup NLTK data
    setup_nltk()
    
    # Load data
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.info('loaded the train and test data')
    logger.info("Train columns: %s", train_data.columns.tolist())
    logger.info("Test columns: %s", test_data.columns.tolist())

    # Process the data
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # Save the processed data
    data_path = os.path.join("data", "processed") 
    os.makedirs(data_path, exist_ok=True)
    train_processed_data.to_csv(os.path.join(data_path, "train_processed_data.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed_data.csv"), index=False)

    print("Preprocessing completed successfully!")
    print(f"Train data saved to: {os.path.join(data_path, 'train_processed_data.csv')}")
    print(f"Test data saved to: {os.path.join(data_path, 'test_processed_data.csv')}")
    print(f"Train data shape: {train_processed_data.shape}")
    print(f"Test data shape: {test_processed_data.shape}")