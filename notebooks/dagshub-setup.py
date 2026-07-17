import mlflow
import dagshub
dagshub.init(repo_owner='Aayush10671', repo_name='emotion-detection', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')


with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)