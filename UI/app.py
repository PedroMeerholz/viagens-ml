import csv
from collections import Counter
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import os

# Caminhos
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset_viagens_brasil.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "decision_tree.pkl"
ENCODER_PATH = Path(__file__).resolve().parents[1] / "models" / "label_encoder.pkl"

MIN_DESTINO_SAMPLE = 100
DESTINO_OUTROS_LABEL = "Outros Destino"

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

def build_form():
    st.title("🧭 Recomendador de Viagens")
    st.write(
        "Preencha seu perfil e preferências de viagem. O modelo irá sugerir um destino com base nas suas escolhas."
    )

    origem_options, destino_labels, destino_decoder = load_city_metadata()

    with st.form("travel_preferences"):
        idade = st.number_input("Idade", min_value=0, max_value=120, value=30, step=1)

        if origem_options:
            origem_select = ["Selecione..."] + origem_options
            cidade_origem = st.selectbox("Cidade de origem", origem_select, index=0)
        else:
            cidade_origem = st.text_input("Cidade de origem")

        custo_desejado = st.number_input(
            "Custo desejado (R$)", min_value=0.0, step=100.0, format="%.2f"
        )

        st.markdown("**Preferências de viagem (0 a 5)**")

        # Usando sliders para representar avaliação de 0 a 5
        prefere_praia = st.slider("Prefere praia", 0, 5, 3)
        prefere_natureza = st.slider("Prefere natureza", 0, 5, 3)
        prefere_cultura = st.slider("Prefere cultura", 0, 5, 3)
        prefere_festas = st.slider("Prefere festas", 0, 5, 3)
        prefere_gastronomia = st.slider("Prefere gastronomia", 0, 5, 3)
        prefere_compras = st.slider("Prefere compras", 0, 5, 3)

        submitted = st.form_submit_button("🎯 Prever destino")

    if submitted:
        if origem_options and cidade_origem == "Selecione...":
            st.warning("Selecione uma cidade de origem.")
            return

        if not origem_options and not cidade_origem.strip():
            st.warning("Informe uma cidade de origem.")
            return

        # Dicionário com os dados do usuário
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

        # Garantir que todos os valores de preferência estão entre 0 e 5
        PREF_COLS = [
            "Prefere_Praia", "Prefere_Natureza", "Prefere_Cultura",
            "Prefere_Festas", "Prefere_Gastronomia", "Prefere_Compras"
        ]

        for col in PREF_COLS:
            val = features[col]
            if not isinstance(val, int) or not (0 <= val <= 5):
                features[col] = 0  # ou np.nan, se preferir avisar

        # Mostra a entrada formatada
        st.subheader("📦 Entrada do usuário:")
        st.json(features)

        # Prepara para o modelo (remove Cidade_Origem se não for usada)
        model_input = pd.DataFrame([features]).drop(columns=["Cidade_Origem"])

        # Carrega e roda o modelo
        if not MODEL_PATH.exists():
            st.error("Modelo não encontrado. Verifique se o arquivo decision_tree.pkl está na pasta correta.")
            return

        with st.spinner("Carregando modelo e realizando previsão..."):
            model = joblib.load(MODEL_PATH)
            pred = model.predict(model_input)[0]
            destino, decoded_with_encoder = decode_destination_label(pred, destino_decoder)
            st.success(f"🌍 Destino previsto: **{destino}**")
            if not decoded_with_encoder and destino != str(pred):
                st.caption("Destino decodificado a partir do dataset base.")
            elif destino == str(pred):
                st.info("Nao foi possivel converter o codigo do destino. Verifique o dataset ou o arquivo de codificacao.")

        if destino_labels:
            st.caption("📍 Destinos considerados pelo modelo:")
            st.write(", ".join(destino_labels))

def main():
    st.set_page_config(
        page_title="Recomendador de Viagens",
        page_icon="✈️",
        layout="centered",
    )
    build_form()


if __name__ == "__main__":
    main()
