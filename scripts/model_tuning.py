from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings("ignore")

import mlflow
import pandas as pd
import json
from sklearn.model_selection import  GridSearchCV
from models.models_config import MODELOS, PARAM_GRIDS
from models.tuning_evaluate import predict, get_classification_report, get_confusion_matrix
from artifacts_generation.plot_results import plot_detailed_overview, plot_general_overview, plot_confusion_matrix


def run_model_tuning(x_train, y_train, x_test, y_test, cidades_originais):
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
            print(train_metrics['train_accuracy_score'])
            mlflow.log_metrics(train_metrics)

            # Faz a previsão e o log das métricas de teste
            prediction, acc_test, precision_test, recall_test, f1_test = predict(clf, x_test, y_test)
            test_metrics = {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
            print(test_metrics['test_accuracy_score'])
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

        # Seleciona os dez melhores modelos baseline
        best_models = sorted(
            baseline_models_info.items(),
            key=lambda item: item[1]['metrics']['test_accuracy_score'],
            reverse=True
        )[:10]

        optimized_report_dir_path = os.environ['OPTIMIZED_RESULTS_DIR']
        # Para cada modelo dos dez melhores, faz a otimização com GridSearchCV
        for model in best_models:
            print(model[0])
            # Instancia o modelo e o grid de parâmetros
            clf = MODELOS[model[0]]
            param_grid = PARAM_GRIDS[model[0]]

            # Faz a otimização do modelo
            grid_search = GridSearchCV(
                clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2, return_train_score=True, pre_dispatch=6
            )
            grid_search.fit(x_train, y_train)

            # Treina o melhor modelo
            best_estimator = grid_search.best_estimator_
            best_estimator.fit(x_train, y_train)

            # Transforma o cv_results_ em um DataFrame
            cv_results = pd.DataFrame(grid_search.cv_results_)

            # Cria e faz o log dos plots de análise de otimização do modelo
            model_name = f'optimized_{model[0]}'
            plot_general_overview(model_name, cv_results, grid_search.best_index_, grid_search.best_score_)
            plot_detailed_overview(model_name, cv_results)

            # Faz o log das métricas de treino
            prediction, acc_train, precision_train, recall_train, f1_train = predict(best_estimator, x_train, y_train)
            # Faz o log das métricas de treino
            train_metrics = {
                'train_accuracy_score': acc_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train
            }
            print(train_metrics['train_accuracy_score'])
            mlflow.log_metrics(train_metrics)

            # Faz a previsão e o log das métricas de teste
            prediction, acc_test, precision_test, recall_test, f1_test = predict(best_estimator, x_test, y_test)
            test_metrics = {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
            print(test_metrics['test_accuracy_score'])
            mlflow.log_metrics(test_metrics)

            # Faz o classification report e faz o log do arquivo
            report = get_classification_report(y_test, prediction)
            report.to_csv(f'{optimized_report_dir_path}/classification_report.csv', index=False)
            mlflow.log_artifact(
                local_path=f'{optimized_report_dir_path}/classification_report.csv',
                artifact_path=model_name
            )

            # Cria a matriz de confusão e faz o log do arquivo
            confusion_matrix = get_confusion_matrix(y_test, prediction)
            plot_confusion_matrix(model_name, confusion_matrix)

            # Faz o log do modelo
            mlflow.sklearn.log_model(
                sk_model=best_estimator,
                name=model_name,
                input_example=x_train[:5]
            )
