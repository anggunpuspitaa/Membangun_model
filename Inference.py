import os
import pandas as pd
import mlflow
import mlflow.sklearn


os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


mlflow.set_tracking_uri("file:./mlruns")

# load model
model = mlflow.sklearn.load_model(
    "runs:/d86f210c6d0246e68fc7e47ab909c745/model"
)

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

pred = model.predict(data)

print("Hasil prediksi:", pred)