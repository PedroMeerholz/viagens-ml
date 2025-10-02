import pandas as pd
from analise_de_dados import run_data_analysis
from transformacao_dos_dados import run_data_preprocessing
from model_tuning import run_model_tuning
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('267582364504963619')


df = pd.read_csv('https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv')
with mlflow.start_run(run_name="Viagens"):
    run_data_analysis(df)
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = run_data_preprocessing(df)
    run_model_tuning(x_train, y_train, x_val, y_val, x_test, y_test, label_encoder)
    mlflow.end_run()
