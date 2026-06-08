import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn

from prometheus_client import start_http_server, Counter, Gauge

# Izinkan MLflow file store
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# Tracking ke folder mlruns lokal
mlflow.set_tracking_uri("file:./mlruns")

# Load model
model = mlflow.sklearn.load_model(
    "runs:/74687eaf673d4571a68fc6606a1311c0/model"
)

# Metrics
request_count = Counter(
    "ml_requests_total",
    "Total prediction requests"
)

prediction_latency = Gauge(
    "prediction_latency_seconds",
    "Prediction latency"
)

prediction_result = Gauge(
    "prediction_result",
    "Prediction result"
)

if __name__ == "__main__":

    print("Prometheus Exporter Berjalan di port 8000")

    start_http_server(8000)

    while True:

        data = pd.DataFrame([{
            "Pclass": 3,
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Sex_male": True,
            "Embarked_Q": False,
            "Embarked_S": True
        }])

        start_time = time.time()

        pred = model.predict(data)[0]

        latency = time.time() - start_time

        request_count.inc()
        prediction_latency.set(latency)
        prediction_result.set(float(pred))

        print(
            f"Prediction={pred}, "
            f"Latency={latency:.5f}"
        )

        time.sleep(5)