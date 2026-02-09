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

    # Transition the newly created model version to the "Staging" stage
    print(f"Attempting to transition model '{model_info.name}' version {model_info.version} to 'Staging'...")
    client = mlflow.tracking.MlflowClient()
    try:
        updated_model_version = client.transition_model_version_stage(
            name="weather-forecaster",
            version=model_info.version,
            stage="Staging"
        )
        print(f"Successfully transitioned model. Current stage: '{updated_model_version.current_stage}'")
        if updated_model_version.current_stage != "Staging":
            print("!!! WARNING: Stage transition was called but the stage did not update to 'Staging'. Check MLflow server permissions and configuration. !!!")
    except Exception as e:
        print(f"!!! ERROR: Failed to transition model stage: {e} !!!")
        # Re-raising the exception to ensure the pipeline fails if staging fails.
        raise



    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)

    run_id = run.info.run_id
    print(f"Model run: {run_id}")
    print(f"Model registered as 'weather-forecaster'")
    print(f"🏃 View run at: http://mlflow:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")
    print(f"🧪 View experiment at: http://mlflow:5000/#/experiments/{run.info.experiment_id}")