import streamlit as st
import requests
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="AI Data Copilot", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Copilot")
st.markdown("Votre assistant intelligent pour l'analyse de données.")

API_URL = "http://localhost:8000/api"

# Initialisation de la session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.sidebar.header("📁 Données")
uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None and st.session_state["session_id"] is None:
    with st.spinner("Analyse du fichier en cours..."):
        # Appel à FastAPI
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                data = response.json()
                st.session_state["session_id"] = data["session_id"]
                st.session_state["metadata"] = data["metadata"]
                st.sidebar.success(f"Fichier {uploaded_file.name} chargé !")
            else:
                st.sidebar.error(f"Erreur: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Erreur de connexion au backend: {str(e)}")

# Si un fichier est chargé, on affiche les métadonnées et le chat
if st.session_state["session_id"]:
    # Onglets
    tab1, tab2 = st.tabs(["💬 Chat Interactif", "🔍 Profiling des données"])
    
    with tab2:
        st.subheader("Aperçu des données")
        meta = st.session_state["metadata"]
        col1, col2 = st.columns(2)
        col1.metric("Lignes", meta["shape"][0])
        col2.metric("Colonnes", meta["shape"][1])
        st.write("Colonnes :", ", ".join(meta["columns"]))
        
        st.subheader("Échantillon (3 lignes)")
        st.dataframe(meta["sample"])

    with tab1:
        col_chat, col_actions = st.columns([3, 1])
        
        with col_actions:
            st.markdown("### ⚡ Actions Rapides")
            quick_actions = [
                "Génère un résumé statistique complet",
                "Quelles sont les corrélations principales ?",
                "Affiche la distribution des colonnes numériques",
                "Détecte les valeurs aberrantes (outliers)",
                "Montre-moi une matrice de corrélation (Heatmap)"
            ]
            
            for action in quick_actions:
                if st.button(action, use_container_width=True):
                    st.session_state["pending_prompt"] = action
                    st.rerun()

        with col_chat:
            # Affichage des messages
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "plots" in msg and msg["plots"]:
                        import plotly.io as pio
                        for p_json in msg["plots"]:
                            try:
                                fig = pio.from_json(p_json)
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                pass
            
            # Gestion du prompt (soit saisi, soit via Quick Action)
            prompt = st.chat_input("Posez une question sur vos données...")
            if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
                prompt = st.session_state["pending_prompt"]
                st.session_state["pending_prompt"] = None
                
            if prompt:
                # Ajouter le message user
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                # Appel API Chat
                with st.chat_message("assistant"):
                    with st.spinner("🧠 Les agents (Planner -> Coder) analysent vos données..."):
                        try:
                            res = requests.post(f"{API_URL}/chat", params={"session_id": st.session_state["session_id"], "prompt": prompt})
                            if res.status_code == 200:
                                api_response = res.json()
                                answer = api_response.get("answer", "Erreur réseau.")
                                plots = api_response.get("plots", [])
                                
                                st.markdown(answer)
                                
                                import plotly.io as pio
                                for p_json in plots:
                                    if p_json:
                                        try:
                                            fig = pio.from_json(p_json)
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as err:
                                            st.error("Impossible de rendre le graphique.")
                                
                                # Sauvegarde dans l'historique
                                st.session_state["messages"].append({
                                    "role": "assistant", 
                                    "content": answer,
                                    "plots": plots
                                })
                            else:
                                st.error(f"Erreur du backend: {res.text}")
                        except Exception as e:
                            st.error(f"Erreur lors de l'appel au chat: {e}")
else:
    st.info("👈 Veuillez uploader un fichier CSV ou Excel dans la barre latérale pour commencer l'expérience Multi-Agents.")
