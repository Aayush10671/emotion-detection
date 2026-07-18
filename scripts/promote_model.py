# promote_model.py
import os
import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # Get credentials from environment
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_password = os.getenv("DAGSHUB_PASSWORD")
    
    if not dagshub_username or not dagshub_password:
        raise EnvironmentError("DAGSHUB_USERNAME and DAGSHUB_PASSWORD must be set")
    
    # Set MLflow environment variables
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_password
    
    # Set tracking URI
    mlflow.set_tracking_uri('https://dagshub.com/Aayush10671/emotion-detection.mlflow')
    
    client = MlflowClient()
    model_name = "sentiment_classifier"
    
    # Get latest version in Staging
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        print("No model found in Staging stage")
        return
    
    latest_staging_version = staging_versions[0].version
    print(f"Found model version {latest_staging_version} in Staging")
    
    # Archive current Production models
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )
        print(f"Archived Production version {version.version}")
    
    # Promote Staging model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_staging_version,
        stage="Production"
    )
    print(f"✅ Model version {latest_staging_version} promoted to Production")

if __name__ == "__main__":
    promote_model()