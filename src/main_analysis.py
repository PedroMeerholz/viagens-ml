from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_processing import obter_dados_processados
from training_pipeline import treinar_e_avaliar
from models_config import MODELOS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle


def gerar_grafico_comparativo(resultados, output_path):
    df_resultados = pd.DataFrame(resultados)

    plt.figure(figsize=(12, 8))
    # Alterado para ordenar e exibir a Acurácia
    sns.barplot(x='acuracia', y='modelo', data=df_resultados.sort_values('acuracia', ascending=False),
                palette='viridis', hue='modelo', legend=False)
    plt.title('Comparação de Acurácia dos Modelos', fontsize=16)
    plt.xlabel('Acurácia', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def executar_analise(modelos_selecionados, usar_tuning, usar_cv, log_queue):
    log_queue.put("Iniciando pipeline de análise...")

    log_queue.put("1. Processando dados...")
    dados = obter_dados_processados()
    if dados is None:
        log_queue.put("ERRO: Falha no processamento de dados. Abortando.")
        return
    log_queue.put("Dados processados com sucesso.")

    X = dados.drop('Cidade_Destino', axis=1)
    y = dados['Cidade_Destino']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    log_queue.put("Divisão de dados: 75% para treino, 25% para teste.")

    colunas_para_escalar = ['Idade', 'Custo_Desejado']
    scaler = StandardScaler()
    scaler.fit(X_train[colunas_para_escalar])
    X_train[colunas_para_escalar] = scaler.transform(X_train[colunas_para_escalar])
    X_test[colunas_para_escalar] = scaler.transform(X_test[colunas_para_escalar])
    log_queue.put("Padronização (scaling) aplicada corretamente.")

    log_queue.put(f"2. Treinando {len(modelos_selecionados)} modelos selecionados...")
    resultados_finais = []

    for i, nome_modelo in enumerate(modelos_selecionados):
        log_queue.put("-" * 40)

        if nome_modelo in MODELOS:
            config_modelo = {'nome': nome_modelo, 'params': MODELOS[nome_modelo].get('default_params', {})}
            resultados = treinar_e_avaliar(X_train, y_train, X_test, y_test, config_modelo, usar_tuning, usar_cv,
                                           log_queue)
            if resultados:
                resultados['modelo'] = nome_modelo
                resultados_finais.append(resultados)
        else:
            log_queue.put(f"AVISO: Modelo '{nome_modelo}' não encontrado na configuração.")

        log_queue.put(('progress', i + 1))

    if resultados_finais:
        log_queue.put("-" * 40)
        log_queue.put("3. Gerando artefatos e identificando o melhor modelo...")

        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        df_final = pd.DataFrame(resultados_finais)

        grafico_path = os.path.join(artifacts_dir, "comparacao_modelos_acuracia.png")
        gerar_grafico_comparativo(resultados_finais, grafico_path)
        log_queue.put(f"Gráfico comparativo de Acurácia salvo em: '{grafico_path}'.")

        resultados_path = os.path.join(artifacts_dir, "resultados_completos.csv")
        df_final_para_salvar = df_final[['modelo', 'acuracia', 'f1_score', 'precisao', 'recall']]
        df_final_para_salvar.to_csv(resultados_path, index=False)
        log_queue.put(f"Resultados completos salvos em: '{resultados_path}'.")

        if not df_final.empty:
            # --- Alteração Principal: Seleção do melhor modelo pela ACURÁCIA ---
            best_model_row = df_final.loc[df_final['acuracia'].idxmax()]
            best_model_name = best_model_row['modelo']

            best_model_result = next(item for item in resultados_finais if item["modelo"] == best_model_name)
            best_model_obj = best_model_result['modelo_obj']

            temp_dir = "temp_artifacts"
            temp_model_path = os.path.join(temp_dir, "best_model.pkl")
            with open(temp_model_path, 'wb') as f:
                pickle.dump(best_model_obj, f)

            log_queue.put(('best_model_path', temp_model_path))

            log_queue.put("\n" + "=" * 50)
            log_queue.put(">>> ANÁLISE CONCLUÍDA <<<")
            # --- Alteração na mensagem de log ---
            log_queue.put(
                f">>> O melhor modelo foi: {best_model_name} com Acurácia de {best_model_row['acuracia']:.4f}")
            log_queue.put(f">>> Modelo pronto para exportação em: {temp_model_path}")
            log_queue.put("=" * 50 + "\n")

    log_queue.put("Pipeline de análise finalizado.")