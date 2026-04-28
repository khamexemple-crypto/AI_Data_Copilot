import streamlit as st
import requests
import pandas as pd
import json
import plotly.io as pio

# Configuration de la page
st.set_page_config(page_title="AI Data Copilot", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Copilot")
st.markdown("Votre assistant intelligent pour l'analyse de données et de documents.")

API_URL = "http://localhost:8000/api"

# Initialisation de la session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.title("🛠️ Menu Principal")
app_mode = st.sidebar.selectbox("Mode de l'application", ["Analyse de Dataset", "File Knowledge Base (RAG)", "Comparaison de Documents", "🧪 Auto Audit"])

# Common Settings
st.sidebar.header("⚙️ Paramètres LLM")
model_selection = st.sidebar.selectbox(
    "Modèle LLM",
    ["Auto recommended", "deepseek-coder-v2:lite", "llama3.2:3b", "qwen2.5:3b", "phi3:mini", "mistral:7b"],
    index=0
)
st.session_state["model_name"] = None if model_selection == "Auto recommended" else model_selection

# ─── MODE DATASET ───
if app_mode == "Analyse de Dataset":
    st.sidebar.header("📁 Données Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

    mode_selection = st.sidebar.radio(
        "Vitesse d'analyse",
        ["🧠 Mode complet", "⚡ Mode rapide"],
        index=0
    )
    st.session_state["analysis_mode"] = "fast" if "rapide" in mode_selection else "deep"

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
                    
                    with st.spinner("🧠 Audit autonome en cours..."):
                        prof_res = requests.post(f"{API_URL}/profile", params={"session_id": st.session_state["session_id"]})
                        if prof_res.status_code == 200:
                            st.session_state["profile_report"] = prof_res.json()
                else:
                    st.sidebar.error(f"Erreur: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Erreur de connexion au backend: {str(e)}")

# ─── MODE RAG ───
else:
    st.sidebar.header("📁 Base de Connaissances")
    kb_files = st.sidebar.file_uploader("Ajouter des documents (PDF, TXT, DOCX...)", accept_multiple_files=True)
    
    if kb_files:
        for f in kb_files:
            if st.sidebar.button(f"🚀 Uploader & Indexer {f.name}"):
                with st.spinner(f"Indexation de {f.name}..."):
                    try:
                        # 1. Upload
                        up_res = requests.post(f"{API_URL}/files/upload", files={"file": (f.name, f.getvalue())}).json()
                        file_id = up_res.get("file_id")
                        # 2. Index
                        idx_res = requests.post(f"{API_URL}/files/index", params={"file_id": file_id, "model_name": st.session_state["model_name"]}).json()
                        st.sidebar.success(f"Indexé : {idx_res.get('indexed_chunks', 0)} fragments.")
                    except Exception as e:
                        st.sidebar.error(f"Erreur: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📚 Documents Indexés**")

    try:
        files_registry = requests.get(f"{API_URL}/files").json()

        if not files_registry:
            st.sidebar.info("Aucun document dans la base.")
        else:
            for fid, fdata in files_registry.items():
                is_indexed = fdata.get("indexed", False)
                icon = "✅" if is_indexed else "⏳"
                label = f"{icon} {fdata['filename']}"

                with st.sidebar.expander(label, expanded=False):

                    if not is_indexed:
                        st.caption("Non indexé — lancez l'indexation ci-dessous.")
                        if st.button("🚀 Lancer l'indexation", key=f"idx_{fid}"):
                            with st.spinner("Indexation..."):
                                requests.post(f"{API_URL}/files/index", params={"file_id": fid})
                            st.rerun()
                        continue

                    # ── Summary ──────────────────────────────────────────────
                    summary = fdata.get("summary") or "Résumé non disponible."
                    st.markdown(f"**📄 Résumé**")
                    st.caption(summary)

                    # ── Tags ─────────────────────────────────────────────────
                    tags = fdata.get("tags", [])
                    if tags:
                        st.markdown("**🏷️ Tags**")
                        tag_html = " ".join(
                            f'<span style="background:#1e3a5f;color:#7dd3fc;'
                            f'padding:2px 8px;border-radius:12px;'
                            f'font-size:0.75rem;margin:2px;display:inline-block">#{t}</span>'
                            for t in tags
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)

                    # ── Key Topics ───────────────────────────────────────────
                    topics = fdata.get("key_topics", [])
                    if topics:
                        st.markdown("**🔑 Sujets clés**")
                        st.markdown("  \n".join(f"• {t}" for t in topics))

                    # ── Suggested questions ───────────────────────────────────
                    questions = fdata.get("suggested_questions", [])
                    if questions:
                        st.markdown("**💬 Questions suggérées**")
                        for q in questions:
                            btn_key = f"q_{fid}_{hash(q) % 99999}"
                            if st.button(f"↗ {q}", key=btn_key, use_container_width=True):
                                st.session_state["pending_rag_prompt"] = q
                                st.rerun()

                    # ── Meta footer ──────────────────────────────────────────
                    chunks  = fdata.get("indexed_chunks", 0)
                    indexed_at = fdata.get("indexed_at", "")
                    if indexed_at:
                        indexed_at = indexed_at[:10]   # date only
                    st.markdown(
                        f"<p style='font-size:0.7rem;color:#6b7280;margin-top:6px'>"
                        f"🗂️ {chunks} fragments · 🕒 {indexed_at}</p>",
                        unsafe_allow_html=True,
                    )

                    # ── Delete button ─────────────────────────────────────────
                    if st.button("🗑️ Supprimer", key=f"del_{fid}", type="secondary"):
                        requests.delete(f"{API_URL}/files/{fid}")
                        st.rerun()

    except Exception as e:
        st.sidebar.warning(f"Impossible de charger les fichiers : {e}")


# ──────────────────────────────────────────────
# MAIN INTERFACE
# ──────────────────────────────────────────────

if app_mode == "Analyse de Dataset":
    if st.session_state["session_id"]:
        tab1, tab2 = st.tabs(["💬 Chat Interactif", "🔍 Profiling & Audit"])
        with tab2:
            if "profile_report" in st.session_state:
                report = st.session_state["profile_report"]
                try:
                    structured = json.loads(report.get("answer", "{}"))
                    st.title("📊 Rapport d'Audit Autonome")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Qualité", f"{int(structured.get('critic', {}).get('confidence', 0)*100)}%")
                    col2.metric("Lignes", st.session_state["metadata"]["shape"][0])
                    col3.metric("Colonnes", st.session_state["metadata"]["shape"][1])
                    st.subheader("💡 Insights Initiaux")
                    for insight in structured.get("analysis", {}).get("insights", []): st.success(insight)
                    if report.get("plots"):
                        for p_json in report["plots"]: st.plotly_chart(pio.from_json(p_json), use_container_width=True)
                except: st.write(report.get("answer"))
        with tab1:
            col_chat, col_actions = st.columns([3, 1])
            with col_actions:
                st.markdown("### ⚡ Actions Rapides")
                for action in ["Résumé statistique", "Corrélations", "Distributions", "Outliers"]:
                    if st.button(action, use_container_width=True):
                        st.session_state["pending_prompt"] = action
                        st.rerun()
            with col_chat:
                for msg in st.session_state["messages"]:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                prompt = st.chat_input("Posez une question sur vos données...")
                if st.session_state.get("pending_prompt"):
                    prompt = st.session_state["pending_prompt"]
                    st.session_state["pending_prompt"] = None
                if prompt:
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        status = st.status("🧠 Analyse en cours...")
                        payload = {"session_id": st.session_state["session_id"], "user_query": prompt, "mode": st.session_state["analysis_mode"], "model_name": st.session_state["model_name"]}
                        res = requests.post(f"{API_URL}/analyze", json=payload).json()
                        status.update(label="✅ Terminé", state="complete")
                        st.markdown(res.get("report", {}).get("final_answer", "Erreur"))
                        st.session_state["messages"].append({"role": "assistant", "content": res.get("report", {}).get("final_answer", "")})
    else:
        st.info("👈 Uploadez un dataset pour commencer.")

elif app_mode == "File Knowledge Base (RAG)":
    st.subheader("💬 Chat avec vos Documents")
    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['filename']}** (Fragment {s['chunk_id']})")
                        st.caption(s.get("text", s.get("excerpt", "")))

    prompt = st.chat_input("Posez une question à votre base de connaissances...")
    if st.session_state.get("pending_rag_prompt"):
        prompt = st.session_state["pending_rag_prompt"]
        st.session_state["pending_rag_prompt"] = None

    if prompt:
        st.session_state["rag_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Recherche et réflexion en cours..."):
                try:
                    res = requests.post(f"{API_URL}/ask-files", params={"query": prompt, "model_name": st.session_state["model_name"]}).json()
                    st.markdown(res.get("answer", "Désolé, je n'ai pas pu générer de réponse."))
                    if res.get("sources"):
                        with st.expander("📚 Sources"):
                            for s in res["sources"]:
                                st.markdown(f"**{s['filename']}** (Fragment {s['chunk_id']})")
                                st.caption(s.get("text", s.get("excerpt", "")))
                    
                    if res.get("limitations"):
                        for lim in res["limitations"]: st.warning(f"Note: {lim}")

                    st.session_state["rag_messages"].append({
                        "role": "assistant", 
                        "content": res.get("answer", ""), 
                        "sources": res.get("sources", [])
                    })
                except Exception as e:
                    st.error(f"Erreur lors de la communication avec le backend: {e}")

elif app_mode == "Comparaison de Documents":
    st.subheader("⚖️ Comparaison Multi-Fichiers")
    
    try:
        files_registry = requests.get(f"{API_URL}/files").json()
        indexed_files = {fid: fdata for fid, fdata in files_registry.items() if fdata.get("indexed")}
        
        if not indexed_files:
            st.warning("Aucun document indexé disponible pour la comparaison.")
        else:
            selected_files = st.multiselect(
                "Sélectionnez les documents à comparer",
                options=list(indexed_files.keys()),
                format_func=lambda fid: indexed_files[fid]["filename"]
            )
            
            question = st.text_input("Sur quoi voulez-vous les comparer ? (ex: Qu'est-ce qui a changé ?)")
            
            if st.button("Lancer la comparaison"):
                if len(selected_files) < 2:
                    st.warning("Veuillez sélectionner au moins deux fichiers.")
                elif not question:
                    st.warning("Veuillez poser une question.")
                else:
                    with st.spinner("Comparaison en cours..."):
                        payload = {
                            "question": question,
                            "file_ids": selected_files,
                            "model_name": st.session_state.get("model_name")
                        }
                        res = requests.post(f"{API_URL}/compare-files", json=payload).json()
                        
                        if res.get("status") == "success":
                            st.success("Comparaison terminée !")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("### ✅ Points Communs")
                                for point in res.get("common_points", []):
                                    st.markdown(f"- {point}")
                                    
                            with col2:
                                st.markdown("### ❌ Différences")
                                for diff in res.get("differences", []):
                                    st.markdown(f"- {diff}")
                                    
                            if res.get("contradictions"):
                                st.markdown("### ⚠️ Contradictions")
                                for contra in res["contradictions"]:
                                    st.error(f"- {contra}")
                                    
                            if res.get("missing_information"):
                                st.markdown("### ℹ️ Informations manquantes")
                                for missing in res["missing_information"]:
                                    st.info(f"- {missing}")
                                    
                            if res.get("sources"):
                                with st.expander("📚 Sources utilisées"):
                                    for s in res["sources"]:
                                        st.markdown(f"**{s['filename']}** (Fragment {s['chunk_id']})")
                                        st.caption(s.get("text", s.get("excerpt", "")))
                                        
                        else:
                            st.error("Impossible de comparer ces documents avec précision.")
                            if res.get("missing_information"):
                                for missing in res["missing_information"]:
                                    st.info(f"- {missing}")
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")

elif app_mode == "🧪 Auto Audit":
    st.title("🧪 Auto-Audit Global")
    st.markdown("Lancez un audit complet croisant les données structurées (dataset) et les informations non structurées (documents).")

    try:
        files_registry = requests.get(f"{API_URL}/files").json()
        indexed_files = {fid: fdata for fid, fdata in files_registry.items() if fdata.get("indexed")}
    except:
        indexed_files = {}

    has_session = st.session_state.get("session_id") is not None
    has_files = len(indexed_files) > 0

    st.write("### État du contexte")
    st.write(f"**Dataset en mémoire:** {'✅ Oui' if has_session else '❌ Non'}")
    st.write(f"**Documents indexés:** {'✅ ' + str(len(indexed_files)) if has_files else '❌ Aucun'}")

    if not has_session and not has_files:
        st.warning("⚠️ Veuillez uploader un dataset ou indexer des documents avant de lancer l'audit.")
    else:
        if st.button("🚀 Lancer l'Auto-Audit", type="primary"):
            with st.spinner("🧠 Exécution de l'audit multi-agents en cours (peut prendre 1-2 minutes)..."):
                payload = {
                    "session_id": st.session_state.get("session_id"),
                    "file_ids": list(indexed_files.keys()),
                    "model_name": st.session_state.get("model_name"),
                    "mode": "deep"
                }
                
                try:
                    res = requests.post(f"{API_URL}/auto-audit", json=payload).json()
                    
                    if res.get("status") == "success":
                        st.success("Audit terminé avec succès !")
                        
                        st.markdown(f"### 📋 Résumé Exécutif")
                        st.info(res.get("summary", ""))
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if res.get("dataset_quality"):
                                st.markdown("### 📊 Qualité des Données")
                                for q in res["dataset_quality"]:
                                    st.write(f"- {q}")
                                    
                            if res.get("risks"):
                                st.markdown("### ⚠️ Risques Identifiés")
                                for r in res["risks"]:
                                    st.error(f"- {r}")
                                    
                        with col2:
                            if res.get("document_findings"):
                                st.markdown("### 📄 Insights Documents")
                                for d in res["document_findings"]:
                                    st.write(f"- {d}")
                                    
                            if res.get("opportunities"):
                                st.markdown("### 💡 Opportunités")
                                for o in res["opportunities"]:
                                    st.success(f"- {o}")
                                    
                        st.markdown("---")
                        
                        if res.get("contradictions"):
                            st.markdown("### ❌ Contradictions")
                            for c in res["contradictions"]:
                                st.warning(f"- {c}")
                                
                        if res.get("recommendations"):
                            st.markdown("### 🎯 Recommandations d'Action")
                            for rec in res["recommendations"]:
                                st.markdown(f"- **{rec}**")
                                
                        if res.get("limitations"):
                            with st.expander("Limitations de l'audit"):
                                for lim in res["limitations"]:
                                    st.write(f"- {lim}")
                                    
                        if res.get("sources"):
                            with st.expander("📚 Sources Utilisées (RAG)"):
                                for s in res["sources"]:
                                    st.markdown(f"**{s['filename']}** (Fragment {s['chunk_id']})")
                                    st.caption(s.get("text", s.get("excerpt", "")))
                    else:
                        st.error("L'audit a échoué.")
                        st.write(res)
                except Exception as e:
                    st.error(f"Erreur lors de la requête: {e}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
