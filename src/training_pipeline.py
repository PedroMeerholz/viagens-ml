import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import os
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, \
    Perceptron
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from models_config import PARAM_GRIDS


def get_model_instance(nome_modelo, params={}):
    model_classes = {
        "RandomForest": RandomForestClassifier, "DecisionTree": DecisionTreeClassifier,
        "LogisticRegression": LogisticRegression, "KNN": KNeighborsClassifier, "SVC": SVC,
        "GradientBoosting": GradientBoostingClassifier, "AdaBoost": AdaBoostClassifier,
        "GaussianNB": GaussianNB, "MLPClassifier": MLPClassifier, "ExtraTrees": ExtraTreesClassifier,
        "LinearSVC": LinearSVC, "RidgeClassifier": RidgeClassifier, "SGDClassifier": SGDClassifier,
        "NuSVC": NuSVC, "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "PassiveAggressiveClassifier": PassiveAggressiveClassifier, "Perceptron": Perceptron,
        "BernoulliNB": BernoulliNB, "NearestCentroid": NearestCentroid,
        "ExtraTreeClassifier": ExtraTreeClassifier,
    }
    return model_classes[nome_modelo](**params)


def treinar_e_avaliar(X_train, y_train, X_test, y_test, config_modelo, usar_tuning, usar_cv, log_queue):
    nome_modelo = config_modelo['nome']
    log_queue.put(f"Processando modelo: {nome_modelo}")

    modelo = get_model_instance(nome_modelo, config_modelo['params'])

    try:
        if usar_tuning:
            log_queue.put(f"Iniciando GridSearch para {nome_modelo}...")
            param_grid = PARAM_GRIDS.get(nome_modelo, {})
            if not param_grid:
                log_queue.put(f"AVISO: Nenhuma grade de hiperparâmetros definida para {nome_modelo}.")
                modelo.fit(X_train, y_train)
            else:
                grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=3, n_jobs=-1,
                                           scoring='f1_weighted')
                grid_search.fit(X_train, y_train)
                modelo = grid_search.best_estimator_
                log_queue.put(f"Melhores parâmetros: {grid_search.best_params_}")
        else:
            log_queue.put(f"Treinando {nome_modelo} com parâmetros padrão...")
            modelo.fit(X_train, y_train)

        if usar_cv:
            log_queue.put("Executando validação cruzada...")
            scores_f1 = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
            log_queue.put(f"F1-Score (CV): Média={np.mean(scores_f1):.4f}, DP={np.std(scores_f1):.4f}")

        y_pred = modelo.predict(X_test)

        # Coleta de todas as métricas
        acuracia = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precisao = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        log_queue.put(f"Acurácia no Teste: {acuracia:.4f}")
        log_queue.put(f"F1-Score no Teste: {f1:.4f}")

        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report_dict).transpose()

        temp_dir = "temp_artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        report_path = os.path.join(temp_dir, f"{nome_modelo}_classification_report.csv")
        df_report.to_csv(report_path)
        log_queue.put(f"Relatório de classificação salvo em: {report_path}")

        # Retorna todas as métricas coletadas
        return {
            'acuracia': acuracia,
            'f1_score': f1,
            'precisao': precisao,
            'recall': recall,
            'modelo_obj': modelo
        }

    except Exception as e:
        log_queue.put(f"ERRO ao treinar o modelo {nome_modelo}: {e}")
        return None