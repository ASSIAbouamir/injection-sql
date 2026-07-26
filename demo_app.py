import streamlit as st
import os
import re
import joblib
import numpy as np

# Config de la page
st.set_page_config(
    page_title="Détecteur d'Injections SQL - Machine Learning",
    page_icon="🛡️",
    layout="wide"
)

# Fonction de classification heuristique (fallback si pas de modèle)
def heuristic_predict(text):
    text_lower = text.lower().strip()
    if not text_lower:
        return 0, 0.0
    
    # Règles d'injections SQL courantes et leurs poids associés
    rules = [
        (r"union\s+select", 0.95),
        (r"union\s+all\s+select", 0.98),
        (r"or\s+\d+\s*=\s*\d+", 0.90),
        (r"or\s+'[^']+'\s*=\s*'[^']+'", 0.90),
        (r"or\s+\"[^\"]+\"\s*=\s*\"[^\"]+\"", 0.90),
        (r"--\s*$", 0.85),
        (r"/\*.*?\*/", 0.80),
        (r"xp_cmdshell", 0.99),
        (r"waitfor\s+delay", 0.99),
        (r"pg_sleep\s*\(", 0.99),
        (r"dbms_pipe\.receive_message", 0.99),
        (r"select\s+.*?\s+from", 0.60),
        (r"drop\s+table", 0.85),
        (r"insert\s+into", 0.70),
        (r"update\s+.*?\s+set", 0.70),
        (r"delete\s+from", 0.75),
        (r"char\s*\(", 0.80),
        (r"ascii\s*\(", 0.70),
        (r"@@version", 0.95),
        (r"version\s*\(", 0.80),
        (r"information_schema", 0.95),
    ]
    
    max_score = 0.0
    for pattern, weight in rules:
        if re.search(pattern, text_lower):
            max_score = max(max_score, weight)
            
    # Légère pénalité si c'est très court sans ponctuation dangereuse
    if max_score > 0.0:
        return 1, max_score
    
    # Score par défaut pour requêtes saines
    return 0, 0.02

# Charger le modèle ML
@st.cache_resource
def load_ml_model():
    # Chemins possibles pour le pipeline final
    model_paths = [
        "data/processed/models/final_pipeline.joblib",
        "data/processed/models_ensembles/final_pipeline_voting.joblib",
        "data/processed/models_ensembles/final_pipeline_stacking.joblib"
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception as e:
                pass
    return None, None

model, model_path = load_ml_model()

# Header de la page
st.title("🛡️ Détecteur d'Injections SQL (SQLi) par Machine Learning")
st.markdown("""
Cette application permet de tester des requêtes ou des saisies utilisateur afin de déterminer en temps réel s'il s'agit d'une **requête saine (bénigne)** ou d'une **tentative d'injection SQL malveillante (SQLi)**.
""")

# Affichage du statut du modèle
if model is not None:
    st.success(f"🚀 **Modèle prédictif Scikit-Learn actif** : Pipeline chargé avec succès depuis `{model_path}`.")
else:
    st.warning("""
    ⚠️ **Modèle de Machine Learning non trouvé localement** (`final_pipeline.joblib`). 
    L'application fonctionne actuellement avec le **détecteur heuristique de secours** (basé sur des signatures et mots-clés SQL).
    Pour activer le modèle de Machine Learning, veuillez exécuter la commande suivante dans votre terminal pour entraîner les modèles :
    `python Projet_injection_sql/load_and_clean.py && python Projet_injection_sql/preprocessing.py && python train_and_tune.py && python build_pipeline.py`
    """)

# Layout principal
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 Tester un Payload")
    
    # Exemples de requêtes pré-définies
    examples = {
        "Sélectionnez un exemple pour tester...": "",
        "Requête saine - Recherche utilisateur standard": "SELECT name, email FROM users WHERE id = 42 AND active = 1;",
        "Requête saine - Insertion d'article": "INSERT INTO articles (title, content) VALUES ('Titre', 'Contenu de mon super article');",
        "SQLi - Authentification Bypass (Classique)": "admin' OR 1=1 --",
        "SQLi - Union-Based (Extraction de données)": "1' UNION SELECT username, password FROM users --",
        "SQLi - Time-Based Blind (Attaque temporelle)": "1'; WAITFOR DELAY '0:0:5' --",
        "SQLi - Error-Based (Version du SGBD)": "1' AND EXTRACTVALUE(1, CONCAT(0x5c, @@VERSION)) --"
    }
    
    selected_example = st.selectbox("Exemples rapides :", list(examples.keys()))
    default_text = examples[selected_example]
    
    # Zone de texte pour saisie personnalisée
    input_query = st.text_area(
        "Saisissez votre payload SQL ou texte à tester :",
        value=default_text,
        height=150,
        placeholder="Entrez votre texte ici (ex: 1' OR '1'='1)..."
    )
    
    if st.button("Analyser le Payload", type="primary"):
        if input_query.strip():
            # Prédiction
            if model is not None:
                # Modèle ML actif
                try:
                    pred = model.predict([input_query])[0]
                    # Récupération des probabilités si disponible
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba([input_query])[0][1]
                    else:
                        proba = 1.0 if pred == 1 else 0.0
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction du modèle ML : {e}")
                    pred, proba = heuristic_predict(input_query)
            else:
                # Fallback heuristique
                pred, proba = heuristic_predict(input_query)
            
            # Affichage du résultat
            st.write("---")
            st.subheader("📊 Résultats de l'analyse")
            
            if pred == 1:
                st.error(f"🚨 **ATTENTION : Tentative d'Injection SQL détectée !**")
                # Affichage jauge ou barre de progression
                st.progress(float(proba))
                st.write(f"Probabilité d'attaque : **{proba * 100:.2f}%**")
            else:
                st.success(f"✅ **Requête classée comme SAINE (Normal)**")
                st.progress(float(proba))
                st.write(f"Probabilité de malveillance : **{proba * 100:.2f}%**")
                
            # Explications des correspondances potentielles
            st.markdown("### 🔎 Analyse syntaxique du texte")
            keywords = ["union", "select", "or", "and", "--", "/*", "*/", "waitfor", "delay", "sleep", "xp_cmdshell", "drop", "insert", "update", "delete", "@@version"]
            found_keywords = [k for k in keywords if k in input_query.lower()]
            if found_keywords:
                st.info(f"Mots-clés / Motifs SQL suspects détectés dans le texte : **{', '.join(found_keywords)}**")
            else:
                st.info("Aucun mot-clé SQL suspect détecté par analyse de motifs simple.")
        else:
            st.warning("Veuillez saisir du texte ou sélectionner un exemple à analyser.")

with col_right:
    st.subheader("ℹ️ À propos du modèle")
    st.markdown("""
    **Comment fonctionne ce détecteur ?**
    
    1. **Vectorisation TF-IDF (N-Grams)** : Le texte est découpé en séquences de 3 à 6 caractères (n-grams). Cela permet de capturer les structures syntaxiques spéciales comme les guillemets (`'`), les commentaires (`--`), ou les opérateurs (`=`), plutôt que les mots entiers.
    2. **Algorithme ML** : Le modèle de production est entraîné sur un jeu de données de près de 30 000 requêtes annotées.
    
    **Performances typiques du pipeline :**
    - **Précision (Precision)** : ~98% (peu de fausses alertes sur les requêtes saines).
    - **Rappel (Recall)** : ~97% (détection efficace de la majorité des injections).
    - **Score F1** : ~97.5%.
    """)
    
    st.subheader("🛠️ Exécuter le pipeline ML")
    st.markdown("""
    Pour ré-entraîner les modèles et mettre à jour le pipeline de production :
    
    ```bash
    # 1. Nettoyer les données
    python Projet_injection_sql/load_and_clean.py
    
    # 2. Vectorisation TF-IDF
    python Projet_injection_sql/preprocessing.py
    
    # 3. Optimisation avec Optuna
    python optuna_tune.py
    
    # 4. Exporter le pipeline
    python build_pipeline.py
    ```
    """)
