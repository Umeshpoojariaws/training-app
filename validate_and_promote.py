import mlflow
from mlflow.tracking import MlflowClient
import os

# Set the MLFLOW_TRACKING_URI environment variable
# This ensures the script connects to the correct MLflow server
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'

MODEL_NAME = "weather-forecaster"

def validate_and_promote():
    client = MlflowClient()
    
    print(f"Starting validation for model: {MODEL_NAME}")

    # 1. Get the Staging Model
    try:
        staging_model = client.get_model_version_by_alias(MODEL_NAME, "staging")
        print(f"Found Staging model: Version {staging_model.version}")
    except Exception:
        print("No model found with alias 'staging'. Nothing to promote.")
        return

    # 2. Get the Production Model
    try:
        production_model = client.get_model_version_by_alias(MODEL_NAME, "production")
        print(f"Found Production model: Version {production_model.version}")
    except Exception:
        print("No model found with alias 'production'. Promoting staging model automatically...")
        promote_to_production(client, staging_model.version)
        return

    # 3. Fetch Metrics for Comparison
    # We need to get the run associated with each model version to access metrics
    try:
        staging_run = client.get_run(staging_model.run_id)
        production_run = client.get_run(production_model.run_id)

        # 'mse' is the metric name logged in train.py
        staging_mse = staging_run.data.metrics.get("mse")
        production_mse = production_run.data.metrics.get("mse")

        if staging_mse is None or production_mse is None:
            print("Could not find 'mse' metric for one or both models. Skipping automated promotion.")
            print(f"Staging MSE: {staging_mse}, Production MSE: {production_mse}")
            return

        print(f"Comparing Metrics (MSE - Lower is better):")
        print(f"  Staging Model (v{staging_model.version}): {staging_mse}")
        print(f"  Production Model (v{production_model.version}): {production_mse}")

        # 4. Compare and Promote
        if staging_mse < production_mse:
            print(f"✅ Staging model performs better! ({staging_mse} < {production_mse})")
            promote_to_production(client, staging_model.version)
        else:
            print(f"❌ Staging model does not perform better. ({staging_mse} >= {production_mse})")
            print("Keeping current production model.")

    except Exception as e:
        print(f"An error occurred during metric comparison: {e}")

def promote_to_production(client, version):
    print(f"Promoting version {version} to 'production' alias...")
    try:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        
        # Update tags for visibility in the UI
        client.set_model_version_tag(MODEL_NAME, version, "status", "production")
        print(f"Successfully promoted version {version} to Production!")
    except Exception as e:
        print(f"Failed to promote model: {e}")

if __name__ == "__main__":
    validate_and_promote()