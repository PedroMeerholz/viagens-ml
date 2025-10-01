from dotenv import load_dotenv
load_dotenv()

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif


def run_data_analysis(df):
    with mlflow.start_run(run_name="Análise de Dados", nested=True):
        head = df.head()
        head.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/sample.csv", index=False)
        mlflow.log_artifact(f"{os.environ['ANALYSIS_RESULTS_DIR']}/sample.csv")

        # Vamos colocar em um .csv com outras informações
        basic_info = dict()
        basic_info['Quantidade de linhas'] = df.shape[0]
        basic_info['Quantidade de colunas'] = df.shape[1]

        # Verificar os tipos de dados
        dtypes = df.dtypes.reset_index()
        dtypes.columns = ['Coluna', 'Tipo']
        dtypes.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/dtypes.csv", index=False)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/dtypes.csv"
        )

        colunas_numericas = ['Idade', 'Custo_Desejado', 'Prefere_Praia', 'Prefere_Natureza',
                            'Prefere_Cultura', 'Prefere_Festas', 'Prefere_Gastronomia', 'Prefere_Compras']

        for coluna in colunas_numericas:
            if coluna in df.columns:
                df[coluna] = pd.to_numeric(df[coluna], errors='coerce').astype('float64')

        null_values_by_column = df.isnull().sum()
        null_values_by_column = null_values_by_column.reset_index()
        null_values_by_column.columns = ['Coluna', 'Qtd. Nulos']
        null_values_by_column.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/null_info.csv", index=False)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/null_info.csv"
        )

        null_rows = null_values_by_column['Qtd. Nulos'].sum()
        basic_info['Quantidade de linhas nulas'] = null_rows
        basic_info_df = pd.DataFrame(list(basic_info.items()), columns=['Informação', 'Valor'])
        basic_info_df.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/basic_info.csv", index=False)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/basic_info.csv"
        )

        duplicated_rows = df.duplicated().sum()
        basic_info['Quantidade de linhas duplicadas'] = duplicated_rows

        cidades_destino_info = df["Cidade_Destino"].value_counts()
        cidades_destino_info.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/contagem_de_classes.csv", index=True)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/contagem_de_classes.csv"
        )

        # Histogramas
        def histogram(x, title, bins=40, kde=True, color='skyblue', path_to_save=''):
            plt.figure(figsize=(10, 5))
            sns.histplot(x, bins=bins, kde=kde, color=color)
            plt.title(title)
            plt.savefig(path_to_save)
            plt.close()

        histogram(df['Idade'], "Distribuição da Idade", bins=40, kde=True, color="skyblue", path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/histograma_idade.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/histograma_idade.png",
            artifact_path="histograma"
        )
        histogram(df['Custo_Desejado'], "Distribuição da Custo Desejado", bins=40, kde=True, color="skyblue", path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/histograma_custo_desejado.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/histograma_custo_desejado.png",
            artifact_path="histograma"
        )
        
        # Boxplots
        def boxplot(x, title, path_to_save):
            plt.figure(figsize=(10, 5))
            plt.title(title)
            sns.boxplot(x=x)
            plt.savefig(path_to_save)
            plt.close()

        boxplot(x=df['Idade'], title='Boxplot Idade', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/boxplot_idade.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/boxplot_idade.png",
            artifact_path="boxplot"
        )
        boxplot(x=df['Custo_Desejado'], title='Boxplot Custo Desejado', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/boxplot_custo_desejado.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/boxplot_custo_desejado.png",
            artifact_path="boxplot"
        )

        # Gráficos de Barra
        def barplot(data, xlabel, ylabel, title, path_to_save):
            plt.figure(figsize=(8,5))
            data.value_counts().sort_index().plot(kind='bar', color='skyblue')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(path_to_save)
            plt.close()

        barplot(df['Prefere_Praia'], 'Prefere_Praia', 'Contagem', 'Distribuição das Preferências por Praia', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_praia.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_praia.png",
            artifact_path="barplot"
        )
        barplot(df['Prefere_Natureza'], 'Prefere_Natureza', 'Contagem', 'Distribuição das Preferências por Natureza', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_natureza.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_natureza.png",
            artifact_path="barplot"
        )
        barplot(df['Prefere_Cultura'], 'Prefere_Cultura', 'Contagem', 'Distribuição das Preferências por Cultura', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_cultura.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_cultura.png",
            artifact_path="barplot"
        )
        barplot(df['Prefere_Festas'], 'Prefere_Festas', 'Contagem', 'Distribuição das Preferências por Festas', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_festas.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_festas.png",
            artifact_path="barplot"
        )
        barplot(df['Prefere_Gastronomia'], 'Prefere_Gastronomia', 'Contagem', 'Distribuição das Preferências por Gastronomia', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_gastronomia.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_gastronomia.png",
            artifact_path="barplot"
        )
        barplot(df['Prefere_Compras'], 'Prefere_Compras', 'Contagem', 'Distribuição das Preferências por Viagem de Compras', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_compras.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_compras.png",
            artifact_path="barplot"
        )
        barplot(df['Cidade_Destino'], 'Cidade_Destino', 'Contagem', 'Distribuição das Preferências por Cidade Destino', path_to_save=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_destino.png")
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/barplot_prefere_destino.png",
            artifact_path="barplot"
        )

        # Importância das Features
        anova_dir = f"{os.environ['ANALYSIS_RESULTS_DIR']}/anova"
        if not os.path.exists(anova_dir):
            os.mkdir(anova_dir)

        # Separar X e y
        le = LabelEncoder()
        df['Cidade_Origem'] = le.fit_transform(df['Cidade_Origem'])
        X = df.drop(['Cidade_Destino'], axis=1)
        y_str = df['Cidade_Destino']

        # Encoder do target
        y_encoded = le.fit_transform(y_str)

        # Tratamento de dados nulos
        X_clean = X.dropna()
        y_clean = y_encoded[X_clean.index]

        # ANOVA F-test scores
        # F-scores,  p-values e mutual information
        f_scores, p_values = f_classif(X_clean, y_clean)
        mi_scores = mutual_info_classif(X_clean, y_clean)

        results = pd.DataFrame({
            'Feature': X.columns,
            'F-Score': f_scores,
            'P-Value': p_values,
            'Mutual Information': mi_scores
        })
        results_sorted = results.sort_values(by='F-Score', ascending=False).reset_index(drop=True)
        results_sorted.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/anova/anova_ftest.csv", index=False)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/anova/anova_ftest.csv",
            artifact_path="anova"    
        )

        class_mapping = dict()
        for i, class_name in enumerate(le.classes_):
            class_mapping[class_name] = i

        class_mapping_df = pd.DataFrame(list(class_mapping.items()), columns=['Cidade_Destino', 'Encoded_Label'])
        class_mapping_df.to_csv(f"{os.environ['ANALYSIS_RESULTS_DIR']}/anova/anova_class_mapping.csv", index=False)
        mlflow.log_artifact(
            local_path=f"{os.environ['ANALYSIS_RESULTS_DIR']}/anova/anova_class_mapping.csv",
            artifact_path="anova"
        )
