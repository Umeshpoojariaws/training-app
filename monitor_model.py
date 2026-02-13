import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
# New 2026 import paths
from evidently.future.datasets import Dataset, DataDefinition, Regression
from evidently.future.report import Report 
from evidently.future.presets import DataDriftPreset # Use the 'future' presets
from google.cloud import storage
from datetime import datetime

# --- Configuration ---
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
MODEL_NAME = "weather-forecaster"
GCS_BUCKET_NAME = "umesh-ml-model-reports"
FEATURES = ['today_temp', 'humidity', 'wind_speed']
TARGET = 'tomorrow_temp'

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False

def monitor_model_drift():
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

    # 2. Create Simulated Current Data
    current_data = reference_data.copy()
    current_data['humidity'] = current_data['humidity'] * 1.2 + 5 
    print("Simulated current production data with drift.")

    # 3. Generate Report (2026 DataDefinition style)
    # print("Generating data drift report with Evidently AI...")
    
    # data_def = DataDefinition(
    #     numerical_columns=FEATURES,
    #     target=TARGET
    # )

    # ref_dataset = Dataset.from_pandas(reference_data[FEATURES + [TARGET]], data_definition=data_def)
    # curr_dataset = Dataset.from_pandas(current_data[FEATURES + [TARGET]], data_definition=data_def)

    # drift_report = Report(metrics=[DataDriftPreset()])
    # drift_report.run(reference_data=ref_dataset, current_data=curr_dataset)

    # 3. Generate Evidently AI Drift Report
    print("Generating data drift report with Evidently AI...")
    
    # 2026 Way: Map columns explicitly via the Regression object
    data_def = DataDefinition(
        regression=[
            Regression(target=TARGET, prediction=None) 
        ],
        numerical_columns=FEATURES
    )

    # In the 'future' API, Dataset objects are required
    ref_dataset = Dataset.from_pandas(reference_data, data_definition=data_def)
    curr_dataset = Dataset.from_pandas(current_data, data_definition=data_def)

    # Use the Report class from evidently.future
    drift_report = Report([
        DataDriftPreset(),
    ])

    # Run the report
    drift_report.run(
        reference_data=ref_dataset,
        current_data=curr_dataset
    )

    # # 4. Save the Report (Using save_html)
    # report_path = "drift_report.html"
    # drift_report.save_html(report_path) # <--- FIXED
    # print(f"✅ Successfully saved data drift report to '{report_path}'")

    # 4. Save the Report
    report_path = "drift_report.html"
    drift_report.save(report_path) # Use save_html, not save()

    # 5. Upload to GCS
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    destination_blob_name = f"drift_report_{timestamp}.html"
    upload_to_gcs(GCS_BUCKET_NAME, report_path, destination_blob_name)

if __name__ == "__main__":
    monitor_model_drift()