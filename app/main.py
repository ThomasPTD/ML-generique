"""
🤖 ML Pipeline — Interface de Prédiction
==========================================
Application Streamlit pour utiliser le meilleur modèle entraîné.
Lance avec : streamlit run app/main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib

# === PAGE CONFIG ===
st.set_page_config(
    page_title="🤖 ML Pipeline — Prédictions",
    page_icon="🤖",
    layout="wide"
)

# === STYLE ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    .error-box {
        padding: 2rem;
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown('<div class="main-header"><h1>🤖 ML Pipeline — Interface de Prédiction</h1></div>',
            unsafe_allow_html=True)

# === LOAD MODEL ===
MODEL_PATH = "models/best_model.joblib"
METADATA_PATH = "models/model_metadata.json"


@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_model()

if model is None or metadata is None:
    st.markdown("""
    <div class="error-box">
        <h2>⚠️ Aucun modèle trouvé</h2>
        <p>Veuillez d'abord entraîner un modèle via le notebook Jupyter (localhost:8888).</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# === MODEL INFO ===
st.sidebar.header("📋 Informations du Modèle")
st.sidebar.markdown(f"**Modèle :** {metadata['model_name']}")
st.sidebar.markdown(f"**Type :** {metadata['task_type'].title()}")
st.sidebar.markdown(f"**Cible :** `{metadata['target_column']}`")
st.sidebar.markdown(f"**Score :** {metadata['best_score']:.4f}")
st.sidebar.markdown(f"**Features :** {len(metadata['feature_columns'])}")
st.sidebar.divider()

# Display benchmark results
if 'all_results' in metadata:
    st.sidebar.header("📊 Benchmark Complet")
    for model_name, scores in metadata['all_results'].items():
        with st.sidebar.expander(model_name):
            for metric, value in scores.items():
                if isinstance(value, (int, float)):
                    st.write(f"**{metric}:** {value:.4f}")

# === TABS ===
tab1, tab2 = st.tabs(["🔮 Prédiction Unitaire", "📁 Prédiction en Lot (CSV)"])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.header("🔮 Prédiction Unitaire")
    st.write("Remplissez les champs ci-dessous pour obtenir une prédiction.")

    feature_cols = metadata['feature_columns']
    numeric_cols = metadata.get('numeric_columns', [])
    categorical_cols = metadata.get('categorical_columns', [])

    # Create input form
    input_data = {}
    cols = st.columns(2)

    for i, feature in enumerate(feature_cols):
        col = cols[i % 2]
        with col:
            if feature in numeric_cols:
                input_data[feature] = st.number_input(
                    f"🔢 {feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{feature}"
                )
            else:
                input_data[feature] = st.text_input(
                    f"🔤 {feature}",
                    value="",
                    key=f"input_{feature}"
                )

    # Predict button
    if st.button("🚀 Prédire", type="primary", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)

            result = prediction[0]
            # Decode label if classification with label encoding
            if metadata.get('label_classes') and metadata['task_type'] == 'classification':
                result = metadata['label_classes'][int(result)]

            st.markdown(f"""
            <div class="prediction-box">
                <h2>✅ Résultat de la Prédiction</h2>
                <h1>{result}</h1>
                <p>Colonne prédite : <b>{metadata['target_column']}</b></p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")

# --- TAB 2: BATCH PREDICTION ---
with tab2:
    st.header("📁 Prédiction en Lot")
    st.write("Uploadez un fichier CSV contenant les mêmes colonnes que les données d'entraînement.")

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"📊 Fichier chargé : {batch_df.shape[0]} lignes × {batch_df.shape[1]} colonnes")
            st.dataframe(batch_df.head())

            # Check that required columns exist
            missing_cols = [c for c in feature_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes dans le CSV : {missing_cols}")
            else:
                if st.button("🚀 Prédire sur tout le CSV", type="primary", use_container_width=True):
                    batch_input = batch_df[feature_cols]
                    predictions = model.predict(batch_input)

                    # Decode labels if needed
                    if metadata.get('label_classes') and metadata['task_type'] == 'classification':
                        predictions = [metadata['label_classes'][int(p)] for p in predictions]

                    batch_df[f"Prédiction_{metadata['target_column']}"] = predictions
                    st.success(f"✅ {len(predictions)} prédictions réalisées !")
                    st.dataframe(batch_df)

                    # Download button
                    csv_output = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv_output,
                        file_name="predictions_output.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
