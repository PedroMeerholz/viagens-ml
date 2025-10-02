import csv
from collections import Counter
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Caminhos
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset_viagens_brasil.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost.pkl"
ENCODER_PATH = Path(__file__).resolve().parents[1] / "models" / "label_encoder.pkl"

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
    if not DATA_PATH.exists():
        return sorted(origem), [], {}

    with DATA_PATH.open(encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            origem_cidade = (row.get("Cidade_Origem") or "").strip()
            destino_cidade = (row.get("Cidade_Destino") or "").strip()
            if origem_cidade:
                origem.add(origem_cidade)
            if destino_cidade:
                destino_counts[destino_cidade] += 1

    destinos_principais = sorted(
        [city for city, count in destino_counts.items() if count >= min_samples]
    )

    possui_outros = any(count < min_samples for count in destino_counts.values())
    if possui_outros and DESTINO_OUTROS_LABEL not in destinos_principais:
        destinos_principais.append(DESTINO_OUTROS_LABEL)

    destino_decoder = build_destino_decoder(destino_counts, min_samples)

    return sorted(origem), destinos_principais, destino_decoder

@st.cache_resource
def load_saved_label_encoder():
    if not ENCODER_PATH.exists():
        return None
    try:
        return joblib.load(ENCODER_PATH)
    except Exception:
        return None

def decode_destination_label(pred_code, destino_decoder):
    label_encoder = load_saved_label_encoder()
    if label_encoder is not None:
        try:
            return label_encoder.inverse_transform([pred_code])[0], True
        except Exception:
            pass

    try:
        numeric_code = int(pred_code)
    except (TypeError, ValueError):
        return str(pred_code), False

    if destino_decoder:
        return destino_decoder.get(numeric_code, str(pred_code)), False

    return str(pred_code), False

# ---------------- INTERFACE PRINCIPAL ----------------
def build_form():
    st.markdown(
        """
        <h1 style="text-align:center; color:white; font-size:42px;">
            🧭 Recomendador de Viagens
        </h1>
        <p style="text-align:center; color:#white; font-size:18px;">
            Descubra o destino ideal com base no seu perfil e preferências ✈️
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    origem_options, destino_labels, destino_decoder = load_city_metadata()

    # Controlador de aba ativa
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "📝 Entrada"

    # Criar menu de navegação manual 
    menu = st.radio(
        "📌 Navegação",
        ["📝 Entrada", "🎯 Resultado", "ℹ️ Detalhes"],
        index=["📝 Entrada", "🎯 Resultado", "ℹ️ Detalhes"].index(st.session_state["active_tab"]),
        horizontal=True
    )

    st.session_state["active_tab"] = menu  # Atualiza aba ativa

    # ------------------ ENTRADA ------------------
    if st.session_state["active_tab"] == "📝 Entrada":
        with st.form("travel_preferences"):
            col1, col2 = st.columns(2)

            with col1:
                idade = st.number_input("👤 Idade", min_value=0, max_value=120, value=30, step=1)
                if origem_options:
                    origem_select = ["Selecione..."] + origem_options
                    cidade_origem = st.selectbox("🏠 Cidade de origem", origem_select, index=0)
                else:
                    cidade_origem = st.text_input("🏠 Cidade de origem")
                custo_desejado = st.number_input("💰 Custo desejado (R$)", min_value=0.0, step=100.0, format="%.2f")

            with col2:
                st.markdown("**✨ Preferências (0 a 5)**")
                prefere_praia = st.slider("🏖️ Praia", 0, 5, 3)
                prefere_natureza = st.slider("🌳 Natureza", 0, 5, 3)
                prefere_cultura = st.slider("🎭 Cultura", 0, 5, 3)
                prefere_festas = st.slider("🎉 Festas", 0, 5, 3)
                prefere_gastronomia = st.slider("🍷 Gastronomia", 0, 5, 3)
                prefere_compras = st.slider("🛍️ Compras", 0, 5, 3)

            submitted = st.form_submit_button("🔍 Calcular Destino")

        if submitted:
            if origem_options and cidade_origem == "Selecione...":
                st.warning("⚠️ Selecione uma cidade de origem.")
                return
            if not origem_options and not cidade_origem.strip():
                st.warning("⚠️ Informe uma cidade de origem.")
                return

            # Features do usuário
            features = {
                "Idade": int(idade),
                "Cidade_Origem": cidade_origem.strip(),
                "Custo_Desejado": float(custo_desejado),
                "Prefere_Praia": int(prefere_praia),
                "Prefere_Natureza": int(prefere_natureza),
                "Prefere_Cultura": int(prefere_cultura),
                "Prefere_Festas": int(prefere_festas),
                "Prefere_Gastronomia": int(prefere_gastronomia),
                "Prefere_Compras": int(prefere_compras),
            }

            st.session_state["features"] = features
            st.session_state["destino_labels"] = destino_labels
            st.session_state["destino_decoder"] = destino_decoder

            # Forçar troca para aba Resultado
            st.session_state["active_tab"] = "🎯 Resultado"
            st.rerun()

    # ------------------ RESULTADO ------------------
    elif st.session_state["active_tab"] == "🎯 Resultado":
        if "features" not in st.session_state:
            st.info("➡️ Preencha o formulário na aba **Entrada**.")
        else:
            features = st.session_state["features"]
            destino_decoder = st.session_state["destino_decoder"]

            model_input = pd.DataFrame([features]).drop(columns=["Cidade_Origem"])
            if not MODEL_PATH.exists():
                st.error("❌ Modelo não encontrado.")
                return

            with st.spinner("🔮 Carregando modelo e prevendo..."):
                model = joblib.load(MODEL_PATH)
                pred = model.predict(model_input)[0]
                destino, _ = decode_destination_label(pred, destino_decoder)

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #2ECC71, #27AE60);
                            padding:20px; border-radius:15px; color:white;
                            text-align:center; font-size:22px; font-weight:bold;
                            box-shadow: 2px 2px 12px rgba(0,0,0,0.3);">
                    🌍 Destino previsto:<br><span style="font-size:28px;">{destino}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------ DETALHES ------------------
    elif st.session_state["active_tab"] == "ℹ️ Detalhes":
        st.markdown("### ℹ️ Sobre este recomendador")
        st.write(
            "Este aplicativo usa um modelo de IA treinado com dados de viagens no Brasil. "
            "Com base nas suas preferências, ele recomenda um destino turístico."
        )

        if "destino_labels" in st.session_state:
            st.markdown("**📍 Destinos considerados pelo modelo:**")
            st.write(", ".join(st.session_state["destino_labels"]))

# ---------------- MAIN ----------------
def main():
    st.set_page_config(page_title="Recomendador de Viagens", page_icon="✈️", layout="wide")

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/69/69906.png", width=80)
    st.sidebar.title("Recomendador de Viagens")
    st.sidebar.markdown("✨ Preencha seus dados e descubra para onde viajar no Brasil!")
    st.sidebar.markdown(
        """
        <div style="background-color:#E74C3C; padding:15px; border-radius:10px;
                    color:white; font-weight:bold; text-align:center;
                    box-shadow:2px 2px 10px rgba(0,0,0,0.3);">
            ⚠️ Este conteúdo é destinado apenas para fins educacionais.<br>
            Os dados exibidos são ilustrativos e podem não corresponder a situações reais.
        </div>
        """,
        unsafe_allow_html=True
    )

    build_form()

if __name__ == "__main__":
    main()