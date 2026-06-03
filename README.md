# Membangun Model Machine Learning

## Deskripsi Proyek

Proyek ini merupakan implementasi machine learning menggunakan dataset Titanic untuk memprediksi kelangsungan hidup penumpang. Proyek mencakup proses pelatihan model, hyperparameter tuning, model tracking menggunakan MLflow, monitoring menggunakan Prometheus, dan visualisasi monitoring menggunakan Grafana.

## Struktur Proyek

* `main.py` : Program utama.
* `modelling.py` : Pelatihan model machine learning.
* `modelling_tuning.py` : Hyperparameter tuning menggunakan GridSearchCV.
* `prometheus_exporter.py` : Exporter metric untuk Prometheus.
* `requirements.txt` : Daftar dependensi proyek.
* `titanic_preprocessed.csv` : Dataset yang digunakan.

## Teknologi yang Digunakan

* Python 3.12
* Scikit-Learn
* MLflow
* Prometheus
* Grafana
* GitHub Actions

## Metrik Monitoring

Metrik yang dipantau menggunakan Prometheus dan Grafana:

* `ml_requests_total`
* `model_accuracy`
* `prediction_latency_seconds`

## Hasil Model

Model terbaik diperoleh menggunakan Random Forest Classifier dengan proses hyperparameter tuning menggunakan GridSearchCV.

## Monitoring Dashboard

Dashboard Grafana digunakan untuk memantau performa model dan metrik sistem secara real-time.

## Cara Menjalankan

### Menjalankan MLflow UI

```bash
py -3.12 -m mlflow ui
```

### Menjalankan Model Serving

```bash
py -3.12 -m mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
```

### Menjalankan Prometheus Exporter

```bash
py -3.12 prometheus_exporter.py
```

### Menjalankan Prometheus

```bash
prometheus.exe
```

### Menjalankan Grafana

Akses:

http://localhost:3000

