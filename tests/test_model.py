# tests/test_model.py
import unittest
import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        
        # Get latest model version
        client = MlflowClient()
        all_versions = client.search_model_versions("name='sentiment_classifier'")
        latest_version = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[0].version
        
        # Load model
        cls.model = mlflow.pyfunc.load_model(f'models:/sentiment_classifier/{latest_version}')
        
        # Load test data
        test_data_path = './data/features/test_bow.csv'
        if not os.path.exists(test_data_path):
            test_data_path = './data/features/train_bow.csv'
        
        test_data = pd.read_csv(test_data_path)
        cls.X_test = test_data.iloc[:, 0:-1].values
        cls.y_test = test_data.iloc[:, -1].values

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        sample_input = self.X_test[0:1]
        prediction = self.model.predict(sample_input)
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        self.assertGreaterEqual(accuracy, 0.40)
        self.assertGreaterEqual(precision, 0.40)
        self.assertGreaterEqual(recall, 0.40)
        self.assertGreaterEqual(f1, 0.40)

if __name__ == "__main__":
    unittest.main()