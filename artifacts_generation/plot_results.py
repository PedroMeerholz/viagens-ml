import math
from os import environ, remove
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow import log_artifact
from dotenv import load_dotenv



def plot_general_overview(model, results, best_index, best_score):
    load_dotenv()
    image_dir_path = environ['PLOT_DIR_PATH']
    
    results['params_str'] = results['params'].apply(
        lambda x: " | ".join([f"{k}={v}" for k, v in x.items()])
    )

    plt.figure(figsize=(10, 6))    # Plotar o resultado médio da validação cruzada
    plt.plot(results['params_str'], results['mean_test_score'], marker='o', linestyle='-', color='blue')
    
    # Destacar a melhor pontuação
    plt.scatter(results.loc[best_index, 'params_str'], best_score, color='red', s=200, label=f'Melhor Score: {best_score:.4f}')

    plt.title("Evolução da Acurácia de Validação Cruzada (Grid Search)")
    plt.xlabel("Combinação de Hiperparâmetros (n_estimators | max_depth)")
    plt.ylabel("Média do Score de Acurácia de Teste (CV)")
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


def plot_detailed_overview(model, results):
    load_dotenv()
    image_dir_path = environ['PLOT_DIR_PATH']
    
    param_cols = [col for col in results.columns if col.startswith('param_')]
    num_params = len(param_cols)

    ncols = 2
    nrows = math.ceil(num_params / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 5))
    axes = axes.flatten()

    for i, param in enumerate(param_cols):
        ax = axes[i]

        # Agrupa por valor do parâmetro e calcula a média da acurácia de teste
        grouped = results.groupby(param)['mean_test_score'].mean().reset_index()
        
        ax.plot(grouped[param], grouped['mean_test_score'], marker='o', linestyle='-')
        ax.set_title(f"Acurácia vs {param.replace('param_', '')}")
        ax.set_xlabel(param.replace('param_', ''))
        ax.set_ylabel("Média da acurácia de teste (CV)")
        ax.grid(True)

    # Oculta os eixos não utilizados
    for j in range(num_params, len(axes)):
        axes[j].axis('off')

    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()
    
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
    plt.show()

    # Salvar e Logar o gráfico
    plot_path = f"{image_dir_path}/confusion_matrix.png"
    fig.savefig(plot_path)
    log_artifact(local_path=plot_path, artifact_path=model)
    plt.close()
    remove(plot_path)
        