import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import mlflow
mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.set_experiment('267582364504963619')

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from models.models_config import MODELOS, PARAM_GRIDS
from models.tuning_evaluate import predict, get_classification_report, get_confusion_matrix
from artifacts_generation.plot_results import plot_detailed_overview, plot_general_overview, plot_confusion_matrix


# Carregar o dataset enviado
df = pd.read_csv('https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv')

# Contagem de quantos 'erro' existem na coluna 'Prefere_Praia'
num_erros = df['Prefere_Praia'].value_counts().get('erro', 0)

# Contagem de qauntos 'cinco' existem na coluna 'Prefere_Cultura'
num_cinco = df['Prefere_Cultura'].value_counts().get('cinco', 0)

# Gera a lista de colunas que começam com "Prefere_"
cols_prefere = [col for col in df.columns if col.startswith("Prefere_")]

for col in cols_prefere:
    # substitui "cinco" por 5 e "erro" por NaN
    df[col] = df[col].replace({'cinco': '5', 'erro': np.nan})

    # Converte para numérico (garante que '1', '2'... virem inteiros)
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Converte para inteiro
    df[col] = df[col].astype('Int32')


def handle_outliers_by_target(df, column, hard_lim_sup: float):
    # Cria uma dataframe vazio para concatenar os dados tratados
    outliers_filled = pd.DataFrame()

    for city in df['Cidade_Destino'].unique():
        # Filtra os dados da cidade com base no df base
        df_city_data = df[df['Cidade_Destino'] == city].copy()

        # Identifica os outliers com base no limite fornecido ou IQR
        outliers = df_city_data[df_city_data[column] > hard_lim_sup]

        # Remove os outliers do df da cidade
        df_city_data = df_city_data.drop(outliers.index)
    
        # Calcula a média da coluna para a cidade
        fill_value = int(df_city_data[column].mean())
    
        # Substitui os valores outliers pela média
        outliers.loc[:, column] = fill_value

        # Concatena os dados tratados
        outliers_filled = pd.concat([outliers_filled, outliers], axis=0)

        # Remove os outliers do df principal
        df = df.drop(outliers.index)
    
    # Adiciona de volta os registros tratados
    df = pd.concat([df, outliers_filled], axis=0).sort_index()
    return df   

df = handle_outliers_by_target(df, 'Idade', hard_lim_sup=100)
df = handle_outliers_by_target(df, 'Custo_Desejado', hard_lim_sup=50000)


def fillna_by_target(df, column, method='mean', default_value=5):
    # Filtra os registros com dados nulos na coluna especificada
    null_values = df[df[column].isnull()]

    # Cria um dataframe vazio para concatenar os dados tratados
    null_values_filled = pd.DataFrame()
    
    for city in df['Cidade_Destino'].unique():
        df_city_data = df[df['Cidade_Destino'] == city]

        # Calcula o valor de preenchimento
        if method == 'mean':
            mean_value = df_city_data[column].mean()
            fill_value = int(mean_value) if pd.notna(mean_value) else default_value

        elif method == 'mode':
            mode_series = df_city_data[column].mode()
            if not mode_series.empty and pd.notna(mode_series.iloc[0]):
                fill_value = mode_series.iloc[0]
            else:
                mean_value = df_city_data[column].mean()
                fill_value = int(mean_value) if pd.notna(mean_value) else default_value
        else:
            raise ValueError("method deve ser 'mean' ou 'mode'")

        # Filtra os registros nulos da cidade (faz uma cópia explícita)
        null_values_city_data = null_values[null_values['Cidade_Destino'] == city].copy()

        # Preenche os valores nulos
        null_values_city_data[column] = null_values_city_data[column].fillna(fill_value)

        # Concatena os dados tratados
        null_values_filled = pd.concat([null_values_filled, null_values_city_data], axis=0)

    # Remove os registros nulos originais
    df = df.drop(null_values.index, axis=0)

    # Adiciona os tratados e ordena
    df = pd.concat([df, null_values_filled], axis=0).sort_index()
    return df


df = fillna_by_target(df, 'Idade', 'mean')
df = fillna_by_target(df, 'Custo_Desejado', 'mean')
df = fillna_by_target(df, 'Prefere_Praia', 'mode')
df = fillna_by_target(df, 'Prefere_Natureza', 'mode')
df = fillna_by_target(df, 'Prefere_Festas', 'mode')
df = fillna_by_target(df, 'Prefere_Gastronomia', 'mode')
df = fillna_by_target(df, 'Prefere_Compras', 'mode')
df = fillna_by_target(df, 'Prefere_Cultura', 'mode')


for city in df['Cidade_Destino'].unique():
    city_data = df[df['Cidade_Destino'] == city].copy()  
    rows = city_data.shape[0]
    if rows < 100:
        city_data.loc[:, 'Cidade_Destino'] = 'Outros Destino'
        df.drop(city_data.index, axis=0, inplace=True)
        df = pd.concat([df, city_data], axis=0)

# Separar a variável alvo
target = "Cidade_Destino"

# Dividir em classes
df_majority = df[df[target] == df[target].value_counts().idxmax()]  # classe majoritária
max_size = df[target].value_counts().max()

# DataFrame final balanceado
df_balanced = pd.DataFrame()
for classe, subset in df.groupby(target):
    # Repete as classes minoritárias até atingir a quantidade da classe majoritária
    df_upsampled = resample(
        subset,
        replace=True,           # permite repetição
        n_samples=max_size,     # iguala ao tamanho da classe majoritária
        random_state=42
    )
    df_balanced = pd.concat([df_balanced, df_upsampled])

# Embaralhar
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

label_encoder = LabelEncoder()

df_balanced['Cidade_Destino'] = label_encoder.fit_transform(df_balanced['Cidade_Destino'])
cidades_originais = label_encoder.inverse_transform(df_balanced['Cidade_Destino'])

X = df_balanced.drop(['Cidade_Origem', 'Cidade_Destino'], axis=1)
y = df_balanced['Cidade_Destino']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del df, df_balanced

with mlflow.start_run(run_name="Baseline models"):
    # Converte o DataFrame de treino para dados do MLFlow e faz o log do arquivo
    train_dataset = mlflow.data.from_pandas(
        pd.concat([x_train, y_train], axis=1), source="train_data.csv", name="viagens-train", targets="Cidade_Destino"
    )
    mlflow.log_input(train_dataset, context='training')

    # Converte o DataFrame de teste para dados do MLFlow e faz o log do arquivo
    test_dataset = mlflow.data.from_pandas(
        pd.concat([x_test, y_test], axis=1), source="test_data.csv", name="viagens-test", targets="Cidade_Destino"
    )
    mlflow.log_input(test_dataset, context='testing')  

    baseline_report_dir_path = os.environ['BASELINE_RESULTS_DIR']

    models = list(MODELOS.keys())[1:2]
    baseline_models_info = dict()
    # Faz o treinamento de cada modelo baseline configurado em MODELOS
    for model in models:
        # Atribui a instância default do modelo
        clf = MODELOS[model]

        # Treina o modelo
        clf.fit(x_train, y_train)
        prediction, acc_train, precision_train, recall_train, f1_train = predict(clf, x_train, y_train)

        # Faz o log das métricas de treino
        mlflow.log_metrics(
            {
                'train_accuracy_score': acc_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train
            }
        )

        # Faz a previsão e o log das métricas de teste
        prediction, acc_test, precision_test, recall_test, f1_test = predict(clf, x_test, y_test)
        mlflow.log_metrics(
            {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
        )

        # Exporta e faz o log do modelo
        mlflow.sklearn.log_model(
            sk_model=clf,
            name=model,
            input_example=x_train[:5]
        )

        # Faz o classification report e faz o log do arquivo
        report = get_classification_report(y_test, prediction)
        report.to_csv(f'{baseline_report_dir_path}/classification_report.csv', index=False)
        mlflow.log_artifact(
            local_path=f'{baseline_report_dir_path}/classification_report.csv',
            artifact_path=model
        )

        # Adiciona as métricas de previsão em um novo dicionário
        baseline_models_info[model] = {
            'params': clf.get_params(),
            'metrics': {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
        }

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
        # Instancia o modelo e o grid de parâmetros
        clf = MODELOS[model[0]]
        param_grid = PARAM_GRIDS[model[0]]

        # Faz a otimização do modelo
        grid_search = GridSearchCV(
            clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2, return_train_score=True
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
        mlflow.log_metrics(
            {
                'train_accuracy_score': acc_train,
                'train_precision': precision_train,
                'train_recall': recall_train,
                'train_f1': f1_train
            }
        )

        # Faz a previsão e o log das métricas de teste
        prediction, acc_test, precision_test, recall_test, f1_test = predict(best_estimator, x_test, y_test)
        mlflow.log_metrics(
            {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
        )

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
