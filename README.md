### Visão Geral

Este projeto constrói um sistema de recomendação de destinos de viagem no Brasil usando técnicas de Machine Learning. O fluxo principal:
- Coleta o dataset público (S3)
- Realiza análise exploratória e geração de artefatos/plots
- Faz tratamento e balanceamento dos dados, além de divisão em treino/validação/teste
- Treina modelos baseline, seleciona os melhores e realiza otimização de hiperparâmetros com Optuna
- Registra métricas, artefatos e modelos no MLflow
- Disponibiliza um app Streamlit (diretório `huggingface/`) para consulta de recomendações

### Principais Tecnologias
- Python 3.12

- scikit-learn

- Optuna (otimização)

- MLflow (rastreamento de experimentos)

- Streamlit (aplicação interativa)

### Estrutura do Projeto (essencial)
- `main.py`: orquestra a pipeline completa (EDA → preprocessamento → treino/otimização)
- `src/analise_de_dados.py`: análise exploratória, relatórios e gráficos
- `src/transformacao_dos_dados.py`: limpeza, tratamento de outliers, preenchimento de nulos, balanceamento e splits
- `src/model_tuning.py`: treino baseline, seleção e otimização (Optuna), logging no MLflow
- `src/tuning_evaluate.py`: funções auxiliares de predição e métricas
- `config/models_config.py`: catálogo de modelos e espaços de busca para o Optuna
- `reports/generate_plots.py`: funções para gráficos de estudos do Optuna e matriz de confusão
- `huggingface/src/streamlit_app.py`: aplicação Streamlit para recomendação
- `data/`: artefatos da análise (csvs e imagens) e resultados processados
- `mlruns/` e `mlartifacts/`: diretórios de execução/artefatos do MLflow

### Dados
- Fonte: `https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv`
- Principais colunas (exemplos): `Idade`, `Cidade_Origem`, `Custo_Desejado`, `Prefere_*`, `Cidade_Destino` (alvo)

### Pipeline
1) Análise de Dados (`run_data_analysis`)
   - Salva amostras (`sample.csv`), tipos (`dtypes.csv`), nulos (`null_info.csv`), contagem de classes, histogramas, boxplots e barplots
   - Calcula ANOVA F-test, p-values e mutual information e salva resultados
2) Pré-processamento (`run_data_preprocessing`)
   - Corrige valores inconsistentes em colunas `Prefere_*`
   - Trata outliers por classe-alvo e gera gráficos de antes/depois
   - Preenche nulos por classe com média/moda
   - Agrega classes raras em `Outros Destino`
   - Balanceia classes por reamostragem e realiza splits (train/val/test)
   - Faz log dos conjuntos no MLflow como inputs
3) Treino e Otimização (`run_model_tuning`)
   - Treina modelos baseline definidos em `config/models_config.py`
   - Loga métricas de treino e teste e exporta modelos para o MLflow
   - Seleciona os melhores por `accuracy` e executa estudo do Optuna
   - Gera e loga gráficos e relatórios (classification report, matriz de confusão, previsões)

### Modelos e Hiperparâmetros
- Catálogo de modelos em `config/models_config.py` (RandomForest, DecisionTree, LogisticRegression, SVC, GradientBoosting, AdaBoost, ExtraTrees, SGDClassifier, BernoulliNB, XGBoost)
- Espaços de busca definidos via funções `PARAM_GRIDS[modelo](trial)` (Optuna)

Resumo dos modelos baseline (visão curta):
- Árvores e ensembles:
  - DecisionTreeClassifier: regras de decisão simples, interpretável.
  - RandomForestClassifier: várias árvores agregadas, mais robusto e reduz overfitting.
  - GradientBoostingClassifier: árvores sequenciais corrigindo erros, costuma performar bem.
  - AdaBoostClassifier: boosting que dá mais peso a erros anteriores.
  - ExtraTreesClassifier: mais aleatoriedade nos splits, rápido e com menor variância.
  - XGBClassifier: implementação otimizada de gradient boosting, alta performance.
- Lineares:
  - LogisticRegression: modela probabilidade de classe, baseline sólido.
  - SGDClassifier: classificador linear otimizado por gradiente estocástico, eficiente em escala.
- SVM:
  - SVC: separa classes com hiperplano ótimo, eficaz em alta dimensionalidade.
- Naive Bayes:
  - BernoulliNB: adequado a variáveis binárias (0/1).

Seleção e otimização:
- Após avaliar os modelos baseline, os 4 melhores por acurácia no teste são selecionados.
- A otimização de hiperparâmetros é feita com Optuna (não GridSearchCV), usando os espaços definidos em `config/models_config.py`. O melhor estimador é re-treinado e avaliado; métricas, relatórios e matrizes de confusão são gerados e logados no MLflow, garantindo modelos performáticos e robustos.

### MLflow
- Configuração em `main.py`:
  - `mlflow.set_tracking_uri('http://localhost:5000')`
  - `mlflow.set_experiment('267582364504963619')`
- Cada etapa (Análise, Tratamento, Treino/Otimização) abre execuções aninhadas
- Artefatos: csvs de análise, imagens, relatórios, métricas e modelos

Para iniciar um servidor local do MLflow (exemplo):
```bash
mlflow server
```
Depois, acesse a UI em `http://localhost:5000`.

### Variáveis de Ambiente
Algumas funções dependem destas variáveis (definir no `.env`):
- `ANALYSIS_RESULTS_DIR`: caminho para salvar artefatos da análise (ex.: `data/analysis`)
- `PLOT_DIR_PATH`: caminho para salvar gráficos de tuning e matriz de confusão (ex.: `data/processed/images`)
- `BASELINE_RESULTS_DIR`: diretório para relatórios/métricas baseline (ex.: `data/processed/model_evaluation/baseline`)
- `OPTIMIZED_RESULTS_DIR`: diretório para relatórios/métricas do melhor modelo (ex.: `data/processed/model_evaluation/optimized`)

Certifique-se de que os diretórios existem ou serão criados pelo seu processo de execução.

### Como Executar (pipeline local)
1) Clonar e instalar dependências
```bash
git clone https://github.com/PedroMeerholz/viagens-ml.git
cd viagens-ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2) Definir `.env` conforme seção de variáveis de ambiente
3) Subir o MLflow server
```bash
mlflow server
```
4) Executar a pipeline
```bash
python main.py
```

### Aplicação Streamlit (Hugging Face)
- Código em `huggingface/src/streamlit_app.py`
- Espera um modelo serializado em `huggingface/models/xgboost.pkl` e consome metadados do dataset S3
- Entradas: idade, cidade de origem, custo desejado e preferências 0–5
- Saída: destino recomendado; exibe também lista de destinos considerados

Para acessar a interface no Hugging Face, acesse:
```bash
https://huggingface.co/spaces/pedromeerholz/viagens-ml
```

### Boas Práticas e Observações
- Os artefatos são salvos e logados no MLflow para rastreabilidade
- O balanceamento por reamostragem ajuda a mitigar classes desbalanceadas
- Valores inconsistentes nas colunas `Prefere_*` são normalizados antes do treino
- O label `Outros Destino` agrupa classes raras para melhorar generalização

### Próximos Passos Sugeridos
- Exportar automaticamente o melhor modelo para o diretório do app (`huggingface/models/`)
