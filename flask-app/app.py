from flask import Flask,request,render_template
import mlflow
import mlflow.pyfunc
from preprocessing_utility import normalize_text

import dagshub
dagshub.init(repo_owner='Aayush10671', repo_name='emotion-detection', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')


app = Flask(__name__)

model_name = "my_model"
model_version = 2

   
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   text = request.form['text']
   #load model from model registry
   text = normalize_text(text)

   
app.run(debug=True)