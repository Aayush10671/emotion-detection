import json
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_PASSWORD")

mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')

def load_model_info(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def register_model(model_name, model_info):
    client = mlflow.tracking.MlflowClient()
    
    # Check if model already exists in registry
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Model '{model_name}' already registered.")
        
        # Register new version
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Move to Staging (NOT Production)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        print(f"Model {model_name} version {model_version.version} registered and moved to Staging")
        
    except Exception as e:
        print(f"Model doesn't exist. Registering new...")
        # If model doesn't exist, register it
        try:
            model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Move to Staging
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"Model {model_name} version {model_version.version} registered and moved to Staging")
        except Exception as reg_error:
            print(f"Could not register model: {reg_error}")

def main():
    model_info = load_model_info('reports/experiment_info.json')
    register_model("sentiment_classifier", model_info)

if __name__ == '__main__':
    main()