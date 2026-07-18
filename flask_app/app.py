# # app.py
# from flask import Flask, request, render_template
# import mlflow
# from mlflow.tracking import MlflowClient
# import pickle
# import pandas as pd
# from preprocessing_utility import normalize_text

# app = Flask(__name__)

# # MLflow setup
# mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

# # Load model
# client = MlflowClient()
# prod_versions = client.get_latest_versions("sentiment_classifier", stages=["Production"])

# if prod_versions:
#     model_version = prod_versions[0].version
# else:
#     all_versions = client.search_model_versions("name='sentiment_classifier'")
#     model_version = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0].version

# model = mlflow.pyfunc.load_model(f"models:/sentiment_classifier/{model_version}")
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html', result=None)

# @app.route('/predict', methods=['POST'])
# def predict():
#     text = request.form['text']
#     text = normalize_text(text)
    
#     # Transform and convert to DataFrame
#     sparse_matrix = vectorizer.transform([text])
#     features_df = pd.DataFrame(
#         sparse_matrix.toarray(),
#         columns=vectorizer.get_feature_names_out()
#     )
    
#     prediction = model.predict(features_df)
#     return render_template('index.html', result=int(prediction[0]))

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


from flask import Flask, request, render_template
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import pandas as pd
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing_utility import normalize_text

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