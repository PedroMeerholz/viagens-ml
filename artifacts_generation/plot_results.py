import math
from os import environ, remove
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow import log_artifact
from dotenv import load_dotenv


def plot_general_overview(model, trials_df, best_trial, best_score):
    load_dotenv()
    image_dir_path = environ['PLOT_DIR_PATH']

    # Extrai apenas as colunas dos parâmetros e o valor objetivo
    param_cols = [col for col in trials_df.columns if col.startswith('params_')]
    value_col = 'value'

    # Cria uma string representando a combinação de hiperparâmetros
    def params_to_str(row):
        return " | ".join([f"{col.replace('params_', '')}={row[col]}" for col in param_cols])

    # Garante que a ordem do eixo x siga a ordem de execução do Optun
    if 'number' in trials_df.columns:
        trials_df_sorted = trials_df.sort_values('number').reset_index(drop=True)
    else:
        trials_df_sorted = trials_df.copy()

    trials_df_sorted['params_str'] = trials_df_sorted.apply(params_to_str, axis=1)

    plt.figure(figsize=(20, 12))
    plt.plot(trials_df_sorted.index, trials_df_sorted[value_col], marker='o', linestyle='-', color='blue')

    # Destaca a melhor pontuação
    best_trial_number = best_trial.number
    best_index = trials_df_sorted[trials_df_sorted['number'] == best_trial_number].index[0]
    plt.scatter(best_index, best_score, color='red', s=200, label=f'Melhor Score: {best_score:.4f}')

    plt.xticks(
        ticks=trials_df_sorted.index,
        labels=trials_df_sorted['params_str'],
        rotation=45,
        ha='right'
    )

    plt.title("Evolução da Métrica de Otimização")
    plt.xlabel("Combinação de Hiperparâmetros")
    plt.ylabel("Acuŕacia da Validação")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Salvar o gráfico como um arquivo temporário
    plot_path = f"{image_dir_path}/grid_search_metrics_evolution.png"
    plt.savefig(plot_path)
    log_artifact(local_path=plot_path, artifact_path=model)
    plt.close()
    remove(plot_path)


def plot_detailed_overview(model, trials_df):
    load_dotenv()
    image_dir_path = environ['PLOT_DIR_PATH']

    # Seleciona apenas as colunas dos hiperparâmetros reais (ignora colunas auxiliares como 'params_str')
    param_cols = [col for col in trials_df.columns if col.startswith('params_') and col != 'params_str']
    num_params = len(param_cols)

    ncols = 2
    nrows = math.ceil(num_params / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
    axes = axes.flatten()

    for i, param in enumerate(param_cols):
        ax = axes[i]
        # Agrupa por valor do parâmetro e calcula a média do valor objetivo ('value')
        grouped = trials_df.groupby(param)['value'].mean().reset_index()
        ax.plot(grouped[param], grouped['value'], marker='o', linestyle='-')
        ax.set_title(f"Métrica vs {param.replace('params_', '')}")
        ax.set_xlabel(param.replace('params_', ''))
        ax.set_ylabel("Acurácia da Validação")
        ax.grid(True)

    # Oculta os eixos não utilizados
    for j in range(num_params, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    # Salvar e Logar o gráfico
    plot_path = f"{image_dir_path}/grid_search_individual_metrics_lines.png"
    fig.savefig(plot_path)
    log_artifact(local_path=plot_path, artifact_path=model)
    plt.close()
    remove(plot_path)


def plot_confusion_matrix(model, confusion_matrix):
    load_dotenv()
    image_dir_path = environ['PLOT_DIR_PATH']

    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(confusion_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, square=True, linewidths=1, linecolor='white')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout(pad=2.0)

    # Salvar e Logar o gráfico
    plot_path = f"{image_dir_path}/confusion_matrix.png"
    fig.savefig(plot_path)
    log_artifact(local_path=plot_path, artifact_path=model)
    plt.close()
    remove(plot_path)
        