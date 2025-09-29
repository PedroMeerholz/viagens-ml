# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
import mlflow
mlflow.set_tracking_uri('http://localhost:5000/')
mlflow.set_experiment('267582364504963619')

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from models.models_config import MODELOS, PARAM_GRIDS
from models.tuning_evaluate import predict, get_classification_report, get_confusion_matrix
from artifacts_generation.plot_results import plot_detailed_overview, plot_general_overview, plot_confusion_matrix
# %%
df = pd.read_csv('https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv')
df.dropna(inplace=True)

# %%
label_encoder = LabelEncoder()

df['Cidade_Destino'] = label_encoder.fit_transform(df['Cidade_Destino'])

cidades_originais = label_encoder.inverse_transform(df['Cidade_Destino'])

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
with mlflow.start_run(run_name="Folder structure test"):
    for model in list(MODELOS.keys())[:1]:
        clf = MODELOS[model]
        param_grid = PARAM_GRIDS[model]

        grid_search = GridSearchCV(
            clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2, return_train_score=True
        )
        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_estimator.fit(x_train, y_train)

        cv_results = pd.DataFrame(grid_search.cv_results_)
        best_index = grid_search.best_index_
        best_score = grid_search.best_score_

        plot_general_overview(model, cv_results, best_index, best_score)
        plot_detailed_overview(model, cv_results)

        prediction, acc_test, precision_test, recall_test, f1_test = predict(best_estimator, x_test, y_test)
        mlflow.log_metrics(
            {
                'test_accuracy_score': acc_test,
                'test_precision': precision_test,
                'test_recall': recall_test,
                'test_f1': f1_test
            }
        )

        report_dir_path = os.environ['EVALUATION_RESULTS_DIR']
        report = get_classification_report(y_test, prediction)
        report.to_csv(f'{report_dir_path}/classification_report.csv', index=False)
        mlflow.log_artifact(
            local_path=f'{report_dir_path}/classification_report.csv',
            artifact_path=f'{model}/classification_report.csv'
        )

        confusion_matrix = get_confusion_matrix(y_test, prediction)
        plot_confusion_matrix(model, confusion_matrix)

        # model_signature = mlflow.models.infer_signature(x_train, clf.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            name=model,
            # signature=model_signature,
            input_example=x_train[:5]
        )
