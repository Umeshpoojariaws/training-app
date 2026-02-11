import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from mlflow.models.signature import infer_signature

# Set the MLFLOW_TRACKING_URI environment variable
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'

# Set the experiment name
mlflow.set_experiment("Weather-Forecast")

with mlflow.start_run() as run:
    # Load the dataset
    df = pd.read_csv('data/weather_data.csv')

    # Define features and target
    features = ['today_temp', 'humidity', 'wind_speed']
    target = 'tomorrow_temp'

    # Split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Force the signature to be float64 to match frontend/backend
    sample_input = X_train.head(1).astype('float64')
    signature = infer_signature(sample_input, model.predict(sample_input))

    # Log the model and get the resulting model version
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="weather-forecast-model",
        signature=signature,
        registered_model_name="weather-forecaster"
    )

    # Set an alias for the staging environment and a tag for UI visibility
    print(f"Attempting to set alias 'staging' for model '{model_info.name}' version {model_info.registered_model_version}...")
    client = mlflow.tracking.MlflowClient()
    try:
        # Set the alias to move this version to 'staging'
        client.set_registered_model_alias(
            name="weather-forecaster",
            alias="staging",
            version=model_info.registered_model_version
        )
        print(f"Successfully set alias 'staging' for model version {model_info.registered_model_version}.")

        # Set a tag for better visibility in the UI
        client.set_model_version_tag(
            name="weather-forecaster",
            version=model_info.registered_model_version,
            key="status",
            value="staging"
        )
        print(f"Successfully set tag 'status: staging' for model version {model_info.registered_model_version}.")

    except Exception as e:
        print(f"!!! ERROR: Failed to set alias or tag for the model: {e} !!!")
        # Re-raising the exception to ensure the pipeline fails if the process fails.
        raise



    # Evaluate the model on the test set
    print("Evaluating model and logging metrics...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    print(f"Logged MSE on test set: {mse}")

    # Save the test data as an artifact for later validation
    print("Saving test data as an artifact...")
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data_path = "test_data.csv"
    test_data.to_csv(test_data_path, index=False)
    mlflow.log_artifact(test_data_path, "validation")
    print(f"Successfully saved '{test_data_path}' to the 'validation' artifact directory.")

    run_id = run.info.run_id
    print(f"Model run: {run_id}")
    print(f"Model registered as 'weather-forecaster'")
    print(f"🏃 View run at: http://mlflow:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")
    print(f"🧪 View experiment at: http://mlflow:5000/#/experiments/{run.info.experiment_id}")