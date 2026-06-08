import time
import requests

from prometheus_client import start_http_server
from prometheus_client import Counter
from prometheus_client import Gauge

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

        payload = {
            "dataframe_records": [
                {
                    "Pclass": 3,
                    "Age": 22,
                    "SibSp": 1,
                    "Parch": 0,
                    "Fare": 7.25,
                    "Sex_male": True,
                    "Embarked_Q": False,
                    "Embarked_S": True
                }
            ]
        }

        start_time = time.time()

        response = requests.post(
            "http://127.0.0.1:5001/invocations",
            json=payload,
            headers={
                "Content-Type": "application/json"
            }
        )

        latency = time.time() - start_time

        pred = response.json()["predictions"][0]

        request_count.inc()
        prediction_latency.set(latency)
        prediction_result.set(float(pred))

        print(
            f"Prediction={pred}, "
            f"Latency={latency:.5f}"
        )

        time.sleep(5)