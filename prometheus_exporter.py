from prometheus_client import start_http_server, Counter, Gauge
import time
import random

request_count = Counter(
    "ml_requests_total",
    "Total requests"
)

model_accuracy = Gauge(
    "model_accuracy",
    "Model accuracy"
)

prediction_latency = Gauge(
    "prediction_latency_seconds",
    "Prediction latency"
)

if __name__ == "__main__":

    start_http_server(8000)

    while True:
        request_count.inc()
        model_accuracy.set(0.82)
        prediction_latency.set(random.uniform(0.01, 0.2))

        time.sleep(5)