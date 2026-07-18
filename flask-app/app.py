from flask import Flask, request, render_template
import mlflow
import mlflow.pyfunc
from preprocessing_utility import normalize_text
import pickle

import dagshub

dagshub.init(repo_owner='Aayush10671', repo_name='emotion-detection', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

app = Flask(__name__)

model_name = "sentiment_classifier"
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)
    features = vectorizer.transform([text])
    prediction = model.predict(features)
    return render_template('index.html', result=int(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True)


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