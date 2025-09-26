# %%
import os
from dotenv import load_dotenv

load_dotenv()

try:
    file_path = os.environ['FILE_PATH']
    print(file_path)
except KeyError:
    print('Variable not set!')

# %%
import mlflow
mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.set_experiment('267582364504963619')

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# %%
le_cidade_destino = LabelEncoder()

df['Cidade_Destino'] = le_cidade_destino.fit_transform(df['Cidade_Destino'])

cidades_originais = le_cidade_destino.inverse_transform(df['Cidade_Destino_encoded'])

# %%
df['Prefere_Praia'] = df['Prefere_Praia'].apply(lambda x: -1 if x == 'erro' else x)
df['Prefere_Praia'] = df['Prefere_Praia'].astype('int64')
df['Prefere_Cultura'] = df['Prefere_Cultura'].apply(lambda x: 5 if x == 'cinco' else x)
df['Prefere_Cultura'] = df['Prefere_Cultura'].astype('int64')
df['Prefere_Compras'] = df['Prefere_Compras'].apply(lambda x: 5 if x == 'cinco' else x)
df['Prefere_Compras'] = df['Prefere_Compras'].astype('int64')

# %%
X = df.drop(['Cidade_Origem', 'Cidade_Destino'], axis=1)
y = df['Cidade_Destino']

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
with mlflow.start_run():

    mlflow.sklearn.autolog()

    clf = RandomForestClassifier(random_state=42)

    clf.fit(x_train, y_train)

    prediction = clf.predict(x_test)

    acc_test = accuracy_score(y_test, prediction)
    precision_test = precision_score(y_test, prediction, average='weighted')
    recall_test = recall_score(y_test, prediction, average='weighted')
    f1_test = f1_score(y_test, prediction, average='weighted')

    mlflow.log_metrics(
        {
            'test_accuracy_score': acc_test,
            'test_precision': precision_test,
            'test_recall': recall_test,
            'test_f1': f1_test
        }
    )

    print(classification_report(y_test, prediction))

    model_signature = mlflow.models.infer_signature(x_train, clf.predict(x_train))
    mlflow.sklearn.log_model(
        sk_model=clf,
        name='rf_model',
        signature=model_signature,
        input_example=x_train[:5]
    )
