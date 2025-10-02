from dotenv import load_dotenv
load_dotenv()

import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def run_data_preprocessing(df):
    with mlflow.start_run(run_name="Tratamento de Dados", nested=True):
        files_path = f"{os.environ['PLOT_DIR_PATH']}"

        # Gera a lista de colunas que começam com "Prefere_"
        cols_prefere = [col for col in df.columns if col.startswith("Prefere_")]
        for col in cols_prefere:
            # substitui "cinco" por 5 e "erro" por NaN
            df[col] = df[col].replace({'cinco': '5', 'erro': np.nan})

            # Converte para numérico (garante que '1', '2'... virem inteiros)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Converte para inteiro
            df[col] = df[col].astype('Int32')

        for col in df.select_dtypes(include=['float64']).columns:
            fig_path = f"{files_path}/{col}_before_outilers_handling.png"
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=df[col], y='Cidade_Destino')
            plt.title(f"Boxplot de {col} - Antes do Tratamento de Outliers")
            plt.savefig(fig_path)
            mlflow.log_artifact(
                local_path=fig_path
            )
            plt.close()

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
        for col in df.select_dtypes(include=['float64']).columns:
            fig_path = f"{files_path}/{col}_after_outilers_handling.png"
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=df[col], y='Cidade_Destino')
            plt.title(f"Boxplot de {col} - Depois do Tratamento de Outliers")
            plt.savefig(fig_path)
            mlflow.log_artifact(
                local_path=fig_path
            )
            plt.close()

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

        X = df_balanced.drop(['Cidade_Origem', 'Cidade_Destino'], axis=1)
        y = df_balanced['Cidade_Destino']
        x_temp, x_val, y_temp, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        x_train, x_test, y_train, y_test = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)
        del df, df_balanced

        # Converte o DataFrame de treino para dados do MLFlow e faz o log do arquivo
        train_dataset = mlflow.data.from_pandas(
            pd.concat([x_train, y_train], axis=1), source="train_data.csv", name="viagens-train", targets="Cidade_Destino"
        )
        mlflow.log_input(train_dataset, context='training')

        val_dataset = mlflow.data.from_pandas(
            pd.concat([x_val, y_val], axis=1), source="val_data.csv", name="viagens-val", targets="Cidade_Destino"
        )
        mlflow.log_input(val_dataset, context='validation')

        # Converte o DataFrame de teste para dados do MLFlow e faz o log do arquivo
        test_dataset = mlflow.data.from_pandas(
            pd.concat([x_test, y_test], axis=1), source="test_data.csv", name="viagens-test", targets="Cidade_Destino"
        )
        mlflow.log_input(test_dataset, context='testing')
        del train_dataset, val_dataset, test_dataset

        return x_train, x_val, x_test, y_train, y_val, y_test, label_encoder
