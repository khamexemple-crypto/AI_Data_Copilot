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
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                data = response.json()
                st.session_state["session_id"] = data["session_id"]
                st.session_state["metadata"] = data["metadata"]
                st.sidebar.success(f"Fichier {uploaded_file.name} chargé !")
                
                # AUTO-PROFILING TRIGGER
                with st.spinner("🧠 Audit autonome en cours..."):
                    prof_res = requests.post(f"{API_URL}/profile", params={"session_id": st.session_state["session_id"]})
                    if prof_res.status_code == 200:
                        st.session_state["profile_report"] = prof_res.json()
                    else:
                        st.sidebar.warning("Échec de l'audit automatique.")
            else:
                st.sidebar.error(f"Erreur: {response.text}")
        except Exception as e:
            st.sidebar.error(f"Erreur de connexion au backend: {str(e)}")

# Si un fichier est chargé, on affiche les métadonnées et le chat
if st.session_state["session_id"]:
    # Onglets
    tab1, tab2 = st.tabs(["💬 Chat Interactif", "🔍 Profiling & Audit"])
    
    with tab2:
        if "profile_report" in st.session_state:
            report = st.session_state["profile_report"]
            try:
                # On parse le structured output
                import json
                structured = json.loads(report.get("answer", "{}"))
                st.title("📊 Rapport d'Audit Autonome")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Qualité", f"{int(structured.get('critic', {}).get('confidence', 0)*100)}%")
                col2.metric("Lignes", st.session_state["metadata"]["shape"][0])
                col3.metric("Colonnes", st.session_state["metadata"]["shape"][1])
                
                st.subheader("💡 Insights Initiaux")
                for insight in structured.get("analysis", {}).get("insights", []):
                    st.success(insight)
                
                if structured.get("analysis", {}).get("anomalies"):
                    st.subheader("⚠️ Alertes de Qualité")
                    for anomaly in structured["analysis"]["anomalies"]:
                        st.warning(anomaly)
                
                if report.get("plots"):
                    st.subheader("📈 Visualisations Clés")
                    import plotly.io as pio
                    for p_json in report["plots"]:
                        fig = pio.from_json(p_json)
                        st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write("Rapport brut :", report.get("answer"))
        else:
            st.info("Chargement du rapport d'audit...")

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
                    status_placeholder = st.status("🧠 Initialisation de l'analyse...", expanded=True)
                    try:
                        status_placeholder.update(label="📅 Planification en cours (Planner Agent)...")
                        res = requests.post(f"{API_URL}/chat", params={"session_id": st.session_state["session_id"], "prompt": prompt})
                        
                        if res.status_code == 200:
                            api_response = res.json()
                            status_placeholder.update(label="✅ Analyse terminée !", state="complete", expanded=False)
                            
                            answer = api_response.get("answer", "Erreur réseau.")
                            plots = api_response.get("plots", [])
                            
                            import json
                            try:
                                parsed_answer = json.loads(answer)
                                if isinstance(parsed_answer, dict) and "final_answer" in parsed_answer:
                                    # Render structured JSON output
                                    st.markdown("### 📋 Plan d'Analyse")
                                    for p in parsed_answer.get("plan", []):
                                        st.markdown(f"- {p}")
                                        
                                    st.markdown("### 🔍 Analyse & Insights")
                                    for i in parsed_answer.get("analysis", {}).get("insights", []):
                                        st.success(i)
                                        
                                    anomalies = parsed_answer.get("analysis", {}).get("anomalies", [])
                                    if anomalies:
                                        st.markdown("#### ⚠️ Anomalies")
                                        for a in anomalies:
                                            st.warning(a)
                                            
                                    preds = parsed_answer.get("predictions", {})
                                    if preds.get("applied"):
                                        st.markdown(f"### 🔮 Prédiction ({preds.get('method')})")
                                        st.info(preds.get("result_summary"))
                                        
                                    critic = parsed_answer.get("critic", {})
                                    with st.expander(f"🤖 Évaluation Critique (Confiance: {int(critic.get('confidence', 0)*100)}%)"):
                                        for issue in critic.get("issues", []):
                                            st.error(f"Problème: {issue}")
                                        for lim in critic.get("limitations", []):
                                            st.write(f"- {lim}")
                                            
                                    st.markdown("### 💡 Conclusion")
                                    st.markdown(parsed_answer.get("final_answer", ""))
                                    
                                    answer_to_save = parsed_answer.get("final_answer", str(answer))
                                else:
                                    st.markdown(answer)
                                    answer_to_save = answer
                            except Exception:
                                st.markdown(answer)
                                answer_to_save = answer
                            
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
                                "content": answer_to_save,
                                "plots": plots
                            })
                        else:
                            status_placeholder.update(label="❌ Erreur Backend", state="error")
                            st.error(f"Erreur du backend: {res.text}")
                    except Exception as e:
                        status_placeholder.update(label="❌ Erreur Connexion", state="error")
                        st.error(f"Erreur lors de l'appel au chat: {e}")
else:
    st.info("👈 Veuillez uploader un fichier CSV ou Excel dans la barre latérale pour commencer l'expérience Multi-Agents.")
