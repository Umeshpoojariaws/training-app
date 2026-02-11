import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# Set the MLFLOW_TRACKING_URI environment variable
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'

MODEL_NAME = "weather-forecaster"
FEATURES = ['today_temp', 'humidity', 'wind_speed']
TARGET = 'tomorrow_temp'

def validate_and_promote():
    client = MlflowClient()
    
    print(f"Starting validation for model: {MODEL_NAME}")

    try:
        staging_model_version = client.get_model_version_by_alias(MODEL_NAME, "staging")
        print(f"Found Staging model: Version {staging_model_version.version}")
    except Exception:
        print("No model found with alias 'staging'. Nothing to promote.")
        return

    # --- New Logic: Fair Comparison ---
    
    # 1. Download the test data artifact from the staging model's run
    try:
        run_id = staging_model_version.run_id
        artifact_path = f"runs:/{run_id}/validation/test_data.csv"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path)
        test_data = pd.read_csv(local_path)
        X_test = test_data[FEATURES]
        y_test = test_data[TARGET]
        print(f"Successfully downloaded test data from run {run_id}")

        # --- FIX: Ensure data types match the model's signature ---
        X_test = X_test.astype('float64')
        print("Casted test data to float64 to match model schema.")

    except Exception as e:
        print(f"Failed to download test data artifact: {e}. Cannot perform fair validation.")
        return

    # 2. Load both models
    try:
        staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@staging")
        print("Successfully loaded staging model.")
    except Exception as e:
        print(f"Failed to load staging model: {e}")
        return

    try:
        production_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")
        print("Successfully loaded production model.")
    except Exception:
        print("No production model found. Promoting staging model automatically.")
        promote_to_production(client, staging_model_version.version)
        return

    # 3. Evaluate both models on the same test set
    staging_pred = staging_model.predict(X_test)
    staging_mse = mean_squared_error(y_test, staging_pred)

    production_pred = production_model.predict(X_test)
    production_mse = mean_squared_error(y_test, production_pred)

    print(f"--- Fair Comparison on New Test Data ---")
    print(f"Staging Model (v{staging_model_version.version}) MSE: {staging_mse}")
    print(f"Production Model MSE: {production_mse}")

    # 4. Compare and Promote
    if staging_mse < production_mse:
        print(f"✅ Staging model performs better! ({staging_mse} < {production_mse})")
        promote_to_production(client, staging_model_version.version)
    else:
        print(f"❌ Staging model does not perform better. ({staging_mse} >= {production_mse})")
        print("Keeping current production model.")


def promote_to_production(client, version):
    print(f"Promoting version {version} to 'production' alias...")
    try:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        client.set_model_version_tag(MODEL_NAME, version, "status", "production")
        print(f"Successfully promoted version {version} to Production!")
    except Exception as e:
        print(f"Failed to promote model: {e}")

if __name__ == "__main__":
    validate_and_promote()