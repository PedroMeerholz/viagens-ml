from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
load_dotenv()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")

import mlflow
import pandas as pd
import json
import optuna
from models.models_config import MODELOS, PARAM_GRIDS
from models.tuning_evaluate import predict, get_classification_report, get_confusion_matrix
from artifacts_generation.plot_results import plot_detailed_overview, plot_general_overview, plot_confusion_matrix


def run_model_tuning(x_train, y_train, x_val, y_val, x_test, y_test):
    with mlflow.start_run(run_name="Treinamento e Otimização de Modelos", nested=True):
        baseline_report_dir_path = os.environ['BASELINE_RESULTS_DIR']

        models = list(MODELOS.keys())
        baseline_models_info = dict()
        # Faz o treinamento de cada modelo baseline configurado em MODELOS
        for model in models:
            print(model)
            # Atribui a instância default do modelo
            clf = MODELOS[model]

            # Treina o modelo
            clf.fit(x_train, y_train)
            prediction, acc_train, precision_train, recall_train, f1_train = predict(clf, x_train, y_train)

            # Faz o log das métricas de treino
            train_metrics = {
                'train_accuracy_score': acc_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train
            }
            mlflow.log_metrics(train_metrics)

            # Faz a previsão e o log das métricas de teste
            prediction, acc_test, precision_test, recall_test, f1_test = predict(clf, x_test, y_test)
            test_metrics = {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
            mlflow.log_metrics(test_metrics)

            # Adiciona as métricas de previsão em um novo dicionário
            baseline_models_info[model] = {
                'params': clf.get_params(),
                'metrics': test_metrics
            }

            # Exporta e faz o log do modelo
            mlflow.sklearn.log_model(
                sk_model=clf,
                name=f'baseline_{model}',
                input_example=x_train[:5]
            )

            # Faz o classification report e faz o log do arquivo
            report = get_classification_report(y_test, prediction)
            report.to_csv(f'{baseline_report_dir_path}/classification_report.csv', index=False)
            mlflow.log_artifact(
                local_path=f'{baseline_report_dir_path}/classification_report.csv',
                artifact_path=model
            )

        # Gera e faz o log do arquivo de métricas
        with open(f"{baseline_report_dir_path}/metrics.json", 'w') as file:
            json.dump(baseline_models_info, file, indent=4)

        mlflow.log_artifact(
            local_path=f'{baseline_report_dir_path}/metrics.json'
        )

        # Seleciona os quatro melhores modelos baseline
        best_models = sorted(
            baseline_models_info.items(),
            key=lambda item: item[1]['metrics']['test_accuracy_score'],
            reverse=True
        )[:4]

        optimized_report_dir_path = os.environ['OPTIMIZED_RESULTS_DIR']
        for model in best_models:
            print(model[0])
            # Instancia o modelo e o grid de parâmetros
            clf = MODELOS[model[0]]
            param_grid = PARAM_GRIDS[model[0]]

            def objective(trial, clf, param_grid, x_train, y_train, x_val, y_val):
                grid = param_grid(trial)
                temp_model = clf.__class__(**grid)
                temp_model.fit(x_train, y_train)
                return temp_model.score(x_val, y_val)

            # Faz a otimização do modelo
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: objective(trial, clf, param_grid, x_train, y_train, x_val, y_val),
                n_trials=20
            )

            # Treina o melhor modelo
            best_params = study.best_params
            best_estimator = clf.__class__(**best_params)
            best_estimator.fit(x_train, y_train)

            # Transforma os trials do Optuna em um DataFrame
            trials_df = study.trials_dataframe()
            
            # Cria e faz o log dos plots de análise de otimização do modelo
            model_name = f'optimized_{model[0]}'

            # TODO Corrigir para utiliar trials_df
            # plot_general_overview(model_name, trials_df, study.best_trial, study.best_value)
            # plot_detailed_overview(model_name, trials_df)

            # Faz o log das métricas de treino
            prediction, acc_train, precision_train, recall_train, f1_train = predict(best_estimator, x_train, y_train)
            train_metrics = {
                'train_accuracy_score': acc_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train
            }
            mlflow.log_metrics(train_metrics)

            # Faz o log das métricas de teste
            prediction, acc_test, precision_test, recall_test, f1_test = predict(best_estimator, x_test, y_test)
            test_metrics = {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
            mlflow.log_metrics(test_metrics)

            # Faz o log do arquivo do classification report
            report = get_classification_report(y_test, prediction)
            report.to_csv(f'{optimized_report_dir_path}/classification_report.csv', index=False)
            mlflow.log_artifact(
                local_path=f'{optimized_report_dir_path}/classification_report.csv',
                artifact_path=model_name
            )

            # Faz o log da matriz de confusão
            confusion_matrix = get_confusion_matrix(y_test, prediction)
            plot_confusion_matrix(model_name, confusion_matrix)

            # Faz o log das previsões de teste
            # prediction_mapped = label_encoder.transform(prediction) # TODO Corrigir mapeamento
            test_predictions_df = x_test.copy()
            test_predictions_df['prediction'] = prediction
            test_predictions_df.to_csv(f'{optimized_report_dir_path}/test_predictions.csv', index=False)
            mlflow.log_artifact(
                local_path=f'{optimized_report_dir_path}/test_predictions.csv',
                artifact_path=model_name
            )

            # Faz o log do modelo
            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                name=model_name,
                input_example=x_train[:5]
            )
