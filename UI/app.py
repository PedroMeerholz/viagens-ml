import csv
from collections import Counter
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# --- CONFIGURAÇÃO DE CAMINHOS (ROBUSTA) ---
try:
    APP_ROOT = Path(__file__).resolve().parents[1]
except (NameError, IndexError):
    APP_ROOT = Path.cwd()

DATA_PATH = 'https://viagens-ml.s3.sa-east-1.amazonaws.com/dataset_viagens_brasil.csv'
MODEL_PATH = APP_ROOT / "models" / "xgboost.pkl"

MIN_DESTINO_SAMPLE = 100
DESTINO_OUTROS_LABEL = "Outros Destino"

# ---------------- FUNÇÕES AUXILIARES ----------------
def normalize_destino_label(city: str, count: int, min_samples: int) -> str:
    city = (city or "").strip()
    if not city:
        return ""
    if city == DESTINO_OUTROS_LABEL:
        return city
    return city if count >= min_samples else DESTINO_OUTROS_LABEL

def build_destino_decoder(destino_counts: Counter, min_samples: int):
    classes = [
        normalize_destino_label(city, count, min_samples)
        for city, count in destino_counts.items()
    ]
    classes = [label for label in classes if label]
    if not classes:
        return {}
    unique_classes = sorted(set(classes))
    return {idx: label for idx, label in enumerate(unique_classes)}

@st.cache_data
def load_city_metadata(min_samples: int = MIN_DESTINO_SAMPLE):
    origem = set()
    destino_counts = Counter()

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if "Cidade_Origem" in df.columns:
        origem.update(df["Cidade_Origem"].dropna().str.strip().unique())
    if "Cidade_Destino" in df.columns:
        destino_counts.update(df["Cidade_Destino"].dropna().str.strip().value_counts().to_dict())

    destinos_principais = sorted(
        [city for city, count in destino_counts.items() if count >= min_samples]
    )

    possui_outros = any(count < min_samples for count in destino_counts.values())
    if possui_outros and DESTINO_OUTROS_LABEL not in destinos_principais:
        destinos_principais.append(DESTINO_OUTROS_LABEL)

    destino_decoder = build_destino_decoder(destino_counts, min_samples)

    return sorted(origem), destinos_principais, destino_decoder

def decode_destination_label(pred_code, destino_decoder):
    try:
        numeric_code = int(pred_code)
    except (TypeError, ValueError):
        return f"Predição Inválida: {pred_code}"

    if destino_decoder:
        return destino_decoder.get(numeric_code, f"Código Desconhecido: {pred_code}")

    return str(pred_code)

# ---------------- PÁGINAS DA APLICAÇÃO ----------------
def pagina_entrada():
    st.markdown(
        """
        <h1 style="text-align:center; color:white; font-size:42px;">
            🧭 Recomendador de Viagens
        </h1>
        <p style="text-align:center; color:white; font-size:18px;">
            Descubra o destino ideal com base no seu perfil e preferências ✈️
        </p>
        <hr style="border-color: #444;">
        """,
        unsafe_allow_html=True
    )

    origem_options, destino_labels, destino_decoder = load_city_metadata()

    with st.form("travel_preferences_form"):
        col1, col2 = st.columns(2)

        with col1:
            idade = st.number_input("👤 Idade", min_value=1, max_value=100, value=30, step=1)
            cidade_origem = st.selectbox("🏠 Cidade de origem", ["Selecione..."] + origem_options, index=0) if origem_options else st.text_input("🏠 Cidade de origem")
            custo_desejado = st.number_input("💰 Custo desejado (R$)", min_value=1100.0, max_value=30000.0, value=1100.0, step=100.0, format="%.2f")

        with col2:
            st.markdown("<p style='text-align: center; font-weight: bold;'>✨ Suas Preferências (0 a 5)</p>", unsafe_allow_html=True)
            prefere_praia = st.slider("🏖️ Praia", 0, 5, 3)
            prefere_natureza = st.slider("🌳 Natureza", 0, 5, 3)
            prefere_cultura = st.slider("🎭 Cultura", 0, 5, 3)
            prefere_festas = st.slider("🎉 Festas", 0, 5, 3)
            prefere_gastronomia = st.slider("🍷 Gastronomia", 0, 5, 3)
            prefere_compras = st.slider("🛍️ Compras", 0, 5, 3)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Calcular Destino Ideal")

    if submitted:
        if (origem_options and cidade_origem == "Selecione...") or (not origem_options and not cidade_origem.strip()):
            st.warning("⚠️ Por favor, selecione ou informe uma cidade de origem.")
            return

        features = {
            "Idade": int(idade), "Cidade_Origem": cidade_origem.strip(), "Custo_Desejado": float(custo_desejado),
            "Prefere_Praia": int(prefere_praia), "Prefere_Natureza": int(prefere_natureza),
            "Prefere_Cultura": int(prefere_cultura), "Prefere_Festas": int(prefere_festas),
            "Prefere_Gastronomia": int(prefere_gastronomia), "Prefere_Compras": int(prefere_compras),
        }

        st.session_state.user_features = features
        st.session_state.destino_labels = destino_labels
        st.session_state.destino_decoder = destino_decoder
        st.session_state.page = "🎯 Resultado"
        st.rerun()

def pagina_resultado():
    st.markdown("<h1 style='text-align:center; color:white;'>🎯 Resultado da Recomendação</h1>", unsafe_allow_html=True)

    if "user_features" not in st.session_state:
        st.info("➡️ Preencha suas preferências na página de Entrada para ver a recomendação.")
        if st.button("Ir para Entrada"):
            st.session_state.page = "📝 Entrada"
            st.rerun()
        return

    if not MODEL_PATH.exists():
        st.error(f"❌ Modelo não encontrado em: {MODEL_PATH}")
        return

    features = st.session_state.user_features
    decoder = st.session_state.destino_decoder
    
    model_input = pd.DataFrame([features]).drop(columns=["Cidade_Origem"])

    with st.spinner("🔮 Analisando suas preferências e calculando o destino..."):
        model = joblib.load(MODEL_PATH)
        prediction_code = model.predict(model_input)[0]
        destino_final = decode_destination_label(prediction_code, decoder)

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #2ECC71, #27AE60); padding: 30px; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-top: 30px;">
            <p style="font-size: 24px; margin-bottom: 10px; font-weight: bold;">Seu destino ideal é:</p>
            <p style="font-size: 42px; font-weight: 800; margin: 0;">{destino_final}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("↩️ Fazer nova consulta"):
        st.session_state.page = "📝 Entrada"
        if 'user_features' in st.session_state:
            del st.session_state['user_features']
        st.rerun()

def pagina_detalhes():
    st.markdown("<h1 style='text-align:center; color:white;'>ℹ️ Detalhes do Projeto</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Sobre o Recomendador")
    st.write(
        """
        Este aplicativo utiliza um modelo de Machine Learning (Árvore de Decisão) treinado com um 
        conjunto de dados de viagens fictícias no Brasil. Ele analisa as preferências de perfil, 
        custo e interesses para sugerir o destino de viagem mais provável para o usuário.
        """
    )

    if "destino_labels" in st.session_state and st.session_state["destino_labels"]:
        st.markdown("### 📍 Destinos Considerados pelo Modelo")
        st.info(", ".join(sorted(st.session_state["destino_labels"])))
    else:
        _, destino_labels, _ = load_city_metadata()
        if destino_labels:
            st.markdown("### 📍 Destinos Considerados pelo Modelo")
            st.info(", ".join(sorted(destino_labels)))

# ---------------- FUNÇÃO PRINCIPAL ----------------
def main():
    st.set_page_config(page_title="Recomendador de Viagens", page_icon="✈️", layout="wide")

    # --- CSS ---
    st.markdown("""
    <style>
        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        div[data-testid="stFormSubmitButton"] > button {
            width: 280px; /* Aumentado */
            height: 70px; /* Aumentado */
            justify-content: center;
            font-size: 24px; /* Aumentado */
            font-weight: bold;
            border-radius: 15px; /* Aumentado */
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: white;
            border: none;
            box-shadow: 0 8px 25px rgba(255, 65, 108, 0.6);
            transition: all 0.3s ease;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 35px rgba(255, 65, 108, 0.8);
        }

        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        /* Ajuste para botões da sidebar e outros botões de ação */
        div[data-testid="stButton"] > button {
            width: 100%; /* Para botões na sidebar com use_container_width */
            height: 70px; /* Aumentado */
            font-size: 24px; /* Aumentado */
            font-weight: bold;
            border-radius: 15px; /* Aumentado */
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: white;
            border: none;
            box-shadow: 0 8px 25px rgba(255, 65, 108, 0.6);
            transition: all 0.3s ease;
            margin-top: 20px; /* Ajuste na margem */
        }
        div[data-testid="stButton"] > button:hover {
            transform: scale(1.05);
            box-shadow: 0 12px 35px rgba(255, 65, 108, 0.8);
        }
        
        /* Específico para botões que não estão na sidebar (para manter a largura fixa) */
        div[data-testid="stAppViewContainer"] div[data-testid="stButton"] > button {
            width: 280px;
        }
    </style>
""", unsafe_allow_html=True)


    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/69/69906.png", width=100)  
        st.title("Recomendador de Viagens")
        st.markdown("✨ Preencha seus dados e descubra para onde viajar no Brasil!")
        st.markdown("---")

        if "page" not in st.session_state:
            st.session_state.page = "📝 Entrada"

        pages = ["📝 Entrada", "🎯 Resultado", "ℹ️ Detalhes"]
        for page in pages:
            if st.button(page, use_container_width=True):
                st.session_state.page = page
                st.rerun()

    if st.session_state.page == "📝 Entrada":
        pagina_entrada()
    elif st.session_state.page == "🎯 Resultado":
        pagina_resultado()
    elif st.session_state.page == "ℹ️ Detalhes":
        pagina_detalhes()

if __name__ == "__main__":
    main()
