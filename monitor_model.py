import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from evidently import Report, Dataset, DataDefinition  # New 2026 components
from evidently.presets import DataDriftPreset
from google.cloud import storage
from datetime import datetime
# --- Configuration ---
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
MODEL_NAME = "weather-forecaster"
GCS_BUCKET_NAME = "umesh-ml-model-reports"
FEATURES = ['today_temp', 'humidity', 'wind_speed']
TARGET = 'tomorrow_temp'

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}."
        )
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False

def monitor_model_drift():
    """
    This script checks for data drift between the model's original test set (reference)
    and a simulated 'current' production dataset.
    """
    client = MlflowClient()
    print("--- Model Drift Monitoring ---")

    # 1. Get the Reference Dataset
    try:
        production_model = client.get_model_version_by_alias(MODEL_NAME, "production")
        run_id = production_model.run_id
        
        artifact_path = f"runs:/{run_id}/validation/test_data.csv"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path)
        
        reference_data = pd.read_csv(local_path)
        print(f"Successfully downloaded reference data from run {run_id}")
    except Exception as e:
        print(f"FATAL: Could not get reference data. Error: {e}")
        return

    # 2. Create a Simulated "Current" Dataset with Drift
    # In a real-world scenario, this data would come from a database or log stream
    # of recent predictions. Here, we simulate drift by altering the humidity.
    current_data = reference_data.copy()
    # Simulating a significant shift in humidity
    current_data['humidity'] = current_data['humidity'] * 1.2 + 5 
    print("Simulated current production data with a drift in 'humidity'.")

    # 3. Generate Evidently AI Drift Report
    print("Generating data drift report with Evidently AI...")
    
    # The column mapping is important for Evidently to distinguish features from the target
    column_mapping = {
        'target': TARGET,
        'prediction': None, # We are only checking for data drift, not prediction drift yet
        'numerical_features': FEATURES
    }

    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    # Run the report
    drift_report.run(
        reference_data=reference_data[FEATURES + [TARGET]],
        current_data=current_data[FEATURES + [TARGET]]
    )

    # 4. Save the Report
    report_path = "drift_report.html"
    drift_report.save_html(report_path)
    print(f"✅ Successfully saved data drift report to '{report_path}'")

    # 5. Upload to GCS
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    destination_blob_name = f"drift_report_{timestamp}.html"
    if upload_to_gcs(GCS_BUCKET_NAME, report_path, destination_blob_name):
        print("🚀 Report successfully uploaded to GCS.")
    else:
        print("⚠️ Failed to upload report to GCS.")


if __name__ == "__main__":
    monitor_model_drift()