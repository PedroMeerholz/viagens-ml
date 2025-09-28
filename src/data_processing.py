import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from urllib.error import URLError

def obter_dados_processados():
    url = 'https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv'

    try:
        df = pd.read_csv(url)
    except URLError:
        print(f"ERRO: Falha de conexão. Não foi possível carregar o dataset da URL: {url}.")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao carregar os dados: {e}")
        return None

    colunas_numericas = [
        'Idade', 'Custo_Desejado', 'Prefere_Praia', 'Prefere_Natureza',
        'Prefere_Cultura', 'Prefere_Festas', 'Prefere_Gastronomia', 'Prefere_Compras'
    ]

    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

    df.dropna(inplace=True)
    df = df[(df['Idade'] >= 18) & (df['Idade'] <= 100)]

    limite = 100
    contagem_cidades = df["Cidade_Destino"].value_counts()
    cidades_para_agrupar = contagem_cidades[contagem_cidades < limite].index
    mapeamento = {cidade: 'Outras cidades' for cidade in cidades_para_agrupar}
    df['Cidade_Destino'] = df['Cidade_Destino'].replace(mapeamento)

    if 'Cidade_Origem' in df.columns:
        le_origem = LabelEncoder()
        df['Cidade_Origem'] = le_origem.fit_transform(df['Cidade_Origem'])

    counts = df['Cidade_Destino'].value_counts()
    if counts.empty:
        print("ERRO: O DataFrame ficou vazio após a limpeza inicial.")
        return None

    target_size = int(counts.median())
    df_balanceado = pd.DataFrame()

    for classe in counts.index:
        df_classe = df[df['Cidade_Destino'] == classe]
        df_classe_resampled = resample(df_classe,
                                       replace=True,
                                       n_samples=target_size,
                                       random_state=42)
        df_balanceado = pd.concat([df_balanceado, df_classe_resampled])

    return df_balanceado