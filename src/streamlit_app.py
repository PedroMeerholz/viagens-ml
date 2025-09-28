import streamlit as st
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import seaborn as sns

# Importando as funções e configurações dos seus outros arquivos
from models_config import PARAM_GRIDS
from main_analysis import executar_analise

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Analisador de Modelos de ML")
st.title("Analisador Interativo de Modelos de Machine Learning 🤖")
st.write(
    "Esta aplicação permite treinar, avaliar e comparar múltiplos modelos de classificação para prever o destino de viagem ideal.")

# --- Inicialização do Estado da Sessão ---
# Usado para manter variáveis entre as interações do usuário
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'best_model_path' not in st.session_state:
    st.session_state.best_model_path = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

# --- Estrutura de Abas ---
tab1, tab2 = st.tabs(["▶️ Treinamento de Modelos", "📊 Análise de Resultados"])

# ==============================================================================
# --- ABA 1: Treinamento de Modelos ---
# ==============================================================================
with tab1:
    # Usamos um formulário para agrupar as seleções e só rodar a análise ao clicar no botão
    with st.form("training_form"):
        st.header("1. Selecione os Modelos")

        model_list = sorted(list(PARAM_GRIDS.keys()))
        num_cols = 5
        cols = st.columns(num_cols)
        modelos_selecionados_vars = {}
        for i, model_name in enumerate(model_list):
            with cols[i % num_cols]:
                modelos_selecionados_vars[model_name] = st.checkbox(model_name, value=True)

        st.header("2. Selecione as Opções de Treinamento")
        col1, col2 = st.columns(2)
        with col1:
            usar_tuning = st.checkbox("Realizar Hiperparametrização (GridSearch)",
                                      help="Busca os melhores parâmetros para cada modelo. Aumenta o tempo de execução.")
        with col2:
            usar_cv = st.checkbox("Usar Validação Cruzada",
                                  help="Avalia a estabilidade do modelo usando 5 folds. Aumenta o tempo de execução.")

        st.header("3. Iniciar a Análise")
        submitted = st.form_submit_button("Executar Análise Completa")

    # --- Lógica de Execução da Análise ---
    if submitted:
        # Reseta o estado da aplicação
        st.session_state.analysis_done = False
        st.session_state.best_model_path = None
        st.session_state.log_messages = []
        st.session_state.df_results = None

        modelos_a_rodar = [name for name, selected in modelos_selecionados_vars.items() if selected]

        if not modelos_a_rodar:
            st.error("Nenhum modelo foi selecionado. Por favor, escolha pelo menos um.")
        else:
            # Placeholders para os elementos da UI que serão atualizados em tempo real
            progress_bar = st.progress(0, text="Análise em progresso...")
            log_container = st.expander("Logs da Execução", expanded=True)
            log_area = log_container.empty()


            # Fila "falsa" para compatibilidade com a função original
            class StreamlitLogQueue:
                def __init__(self):
                    self.messages = []

                def put(self, message):
                    if isinstance(message, tuple) and message[0] == 'progress':
                        total_modelos = len(modelos_a_rodar)
                        progresso_atual = message[1]
                        percentual = int((progresso_atual / total_modelos) * 100)
                        progress_bar.progress(percentual,
                                              text=f"Progresso: {percentual}% ({progresso_atual}/{total_modelos} modelos)")
                    elif isinstance(message, tuple) and message[0] == 'best_model_path':
                        st.session_state.best_model_path = message[1]
                    else:
                        self.messages.append(str(message))
                        # Atualiza a área de log em tempo real
                        log_area.code('\n'.join(self.messages))


            st_log_queue = StreamlitLogQueue()

            # Executa a análise (esta é uma operação síncrona/bloqueante)
            try:
                executar_analise(modelos_a_rodar, usar_tuning, usar_cv, st_log_queue)
                st.session_state.analysis_done = True
                st.session_state.log_messages = st_log_queue.messages  # Salva os logs finais
                st.success("Análise concluída com sucesso!")
                st.balloons()
            except Exception as e:
                st.error(f"Ocorreu um erro crítico durante a análise: {e}")

    # --- Exibição dos Resultados Pós-Análise ---
    if st.session_state.analysis_done:
        st.header("4. Resultados da Análise")

        # Exibe o botão de download se o modelo foi salvo
        if st.session_state.best_model_path and os.path.exists(st.session_state.best_model_path):
            with open(st.session_state.best_model_path, "rb") as f:
                st.download_button(
                    label="📥 Baixar Melhor Modelo (.pkl)",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.warning("Não foi possível encontrar o arquivo do melhor modelo para download.")

        with st.expander("Ver Logs Finais da Execução", expanded=False):
            st.code('\n'.join(st.session_state.log_messages))

# ==============================================================================
# --- ABA 2: Análise de Resultados ---
# ==============================================================================
with tab2:
    st.header("Visualização Comparativa das Métricas")

    # Botão para carregar os dados
    if st.button("Carregar e Exibir Resultados da Última Análise"):
        results_path = "artifacts/resultados_completos.csv"
        if os.path.exists(results_path):
            st.session_state.df_results = pd.read_csv(results_path)
            st.success("Resultados carregados com sucesso!")
        else:
            st.session_state.df_results = None
            st.warning("Arquivo de resultados não encontrado. Execute uma análise na primeira aba.")

    # Se os dados estiverem carregados, exibe os controles e o gráfico
    if st.session_state.df_results is not None:
        chart_type = st.radio(
            "Selecione o Tipo de Gráfico:",
            ("Barras", "Pontos/Linhas"),
            horizontal=True
        )

        df = st.session_state.df_results.copy()
        df.rename(columns={'acuracia': 'Acurácia', 'f1_score': 'F1-Score',
                           'precisao': 'Precisão', 'recall': 'Recall'}, inplace=True)

        df_melted = df.melt(id_vars='modelo', var_name='Métrica', value_name='Valor')
        order = df.sort_values('Acurácia', ascending=False).modelo

        fig, ax = plt.subplots(figsize=(12, 10))

        if chart_type == "Barras":
            sns.barplot(data=df_melted, y='modelo', x='Valor', hue='Métrica', ax=ax, order=order)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
        else:
            sns.pointplot(data=df_melted, y='modelo', x='Valor', hue='Métrica', ax=ax, order=order, dodge=True)
            ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_title('Comparação de Métricas dos Modelos', fontsize=16)
        ax.set_ylabel('Modelo', fontsize=12)
        ax.set_xlabel('Pontuação', fontsize=12)
        ax.set_xlim(0, max(1.0, df_melted['Valor'].max() * 1.1))

        fig.tight_layout()
        st.pyplot(fig)