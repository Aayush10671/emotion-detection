# app.py
from flask import Flask, request, render_template
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import pandas as pd

# data preprocessing
import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score , precision_score , recall_score,f1_score
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



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


def normalize_text(text):
   
        text = lower_case(text)
        text = remove_stop_words(text)
        text = removing_numbers(text)
        text = removing_punctuations(text)
        text = removing_urls(text)
        text = lemmatization(text)
        return text
  

  

app = Flask(__name__)

# MLflow setup
mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

# Load model
client = MlflowClient()
prod_versions = client.get_latest_versions("sentiment_classifier", stages=["Production"])

if prod_versions:
    model_version = prod_versions[0].version
else:
    all_versions = client.search_model_versions("name='sentiment_classifier'")
    model_version = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0].version

model = mlflow.pyfunc.load_model(f"models:/sentiment_classifier/{model_version}")
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)
    
    # Transform and convert to DataFrame
    sparse_matrix = vectorizer.transform([text])
    features_df = pd.DataFrame(
        sparse_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    
    prediction = model.predict(features_df)
    return render_template('index.html', result=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


















# from fastapi import FastAPI, Request, Form
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse
# import mlflow
# import mlflow.pyfunc
# from preprocessing_utility import normalize_text
# import pickle

# import dagshub

# dagshub.init(repo_owner='Aayush10671', repo_name='emotion-detection', mlflow=True)
# mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# model_name = "sentiment_classifier"
# model_version = 2
# model_uri = f"models:/{model_name}/{model_version}"
# model = mlflow.pyfunc.load_model(model_uri)

# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))


# @app.get('/', response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse(
#         "index.html", {"request": request, "result": None, "text": ""}
#     )


# @app.post('/predict', response_class=HTMLResponse)
# def predict(request: Request, text: str = Form(...)):
#     cleaned = normalize_text(text)
#     features = vectorizer.transform([cleaned])
#     prediction = model.predict(features)
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request, "result": int(prediction[0]), "text": text}
#     )