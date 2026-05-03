import streamlit as st
import requests
import pandas as pd
import json
import plotly.io as pio
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(page_title="AI Data Copilot", layout="wide", page_icon="🤖")

st.title("🤖 AI Data Copilot")
st.markdown("Votre assistant intelligent pour l'analyse de données et de documents.")

API_URL = "http://localhost:8000/api"


def _clean_params(params):
    if not params:
        return params
    return {key: value for key, value in params.items() if value is not None}


def api_json(method: str, url: str, timeout: int = 120, **kwargs):
    kwargs["params"] = _clean_params(kwargs.get("params"))
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Backend inaccessible: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, list):
            detail = " | ".join(str(item) for item in detail)
        raise RuntimeError(str(detail or response.text or f"HTTP {response.status_code}"))

    return payload


def reset_dataset_session():
    st.session_state["session_id"] = None
    st.session_state["metadata"] = None
    st.session_state["messages"] = []
    st.session_state.pop("profile_report", None)


def render_voice_player(text: str, language: str = "fr-FR", rate: float = 1.0, pitch: float = 1.0):
    player_id = f"voice_{abs(hash(text)) % 999999}"
    safe_text = json.dumps(text or "")
    safe_lang = json.dumps(language or "fr-FR")
    components.html(
        f"""
        <div style="font-family: system-ui, -apple-system, Segoe UI, sans-serif; display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
          <button onclick="{player_id}_speak()" style="padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;background:#111827;color:white;cursor:pointer;">Lire</button>
          <button onclick="{player_id}_pause()" style="padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;background:white;color:#111827;cursor:pointer;">Pause</button>
          <button onclick="{player_id}_resume()" style="padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;background:white;color:#111827;cursor:pointer;">Reprendre</button>
          <button onclick="{player_id}_stop()" style="padding:8px 12px;border:1px solid #d1d5db;border-radius:6px;background:white;color:#991b1b;cursor:pointer;">Stop</button>
          <span id="{player_id}_status" style="font-size:13px;color:#4b5563;"></span>
        </div>
        <script>
        const {player_id}_text = {safe_text};
        const {player_id}_lang = {safe_lang};
        function {player_id}_setStatus(value) {{
          document.getElementById("{player_id}_status").innerText = value;
        }}
        function {player_id}_speak() {{
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance({player_id}_text);
          utterance.lang = {player_id}_lang;
          utterance.rate = {rate};
          utterance.pitch = {pitch};
          utterance.onstart = () => {player_id}_setStatus("Lecture en cours");
          utterance.onend = () => {player_id}_setStatus("Termine");
          utterance.onerror = () => {player_id}_setStatus("Lecture indisponible sur ce navigateur");
          window.speechSynthesis.speak(utterance);
        }}
        function {player_id}_pause() {{
          window.speechSynthesis.pause();
          {player_id}_setStatus("Pause");
        }}
        function {player_id}_resume() {{
          window.speechSynthesis.resume();
          {player_id}_setStatus("Lecture en cours");
        }}
        function {player_id}_stop() {{
          window.speechSynthesis.cancel();
          {player_id}_setStatus("Stop");
        }}
        </script>
        """,
        height=64,
    )

# Initialisation de la session state
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "metadata" not in st.session_state:
    st.session_state["metadata"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "voice_presentation" not in st.session_state:
    st.session_state["voice_presentation"] = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.title("🛠️ Menu Principal")
app_mode = st.sidebar.selectbox(
    "Mode de l'application",
    [
        "Analyse de Dataset",
        "File Knowledge Base (RAG)",
        "Comparaison de Documents",
        "🧪 Auto Audit",
        "Storage Hub",
        "Présentation Vocale",
    ],
)

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

    if st.session_state["session_id"]:
        filename = st.session_state.get("metadata", {}).get("filename", "dataset chargé")
        st.sidebar.caption(f"Dataset actif : {filename}")
        if st.sidebar.button("Changer de dataset", use_container_width=True):
            reset_dataset_session()
            st.rerun()

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
                data = api_json("POST", f"{API_URL}/upload", files=files, timeout=180)
                st.session_state["session_id"] = data["session_id"]
                st.session_state["metadata"] = data["metadata"]
                st.sidebar.success(f"Fichier {uploaded_file.name} chargé.")

                with st.spinner("🧠 Audit autonome en cours..."):
                    try:
                        st.session_state["profile_report"] = api_json(
                            "POST",
                            f"{API_URL}/profile",
                            params={"session_id": st.session_state["session_id"]},
                            timeout=240,
                        )
                    except Exception as profile_error:
                        st.sidebar.warning(f"Dataset chargé, mais le profiling a échoué : {profile_error}")
            except Exception as e:
                st.sidebar.error(f"Upload impossible : {str(e)}")

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
                        up_res = api_json(
                            "POST",
                            f"{API_URL}/files/upload",
                            files={"file": (f.name, f.getvalue())},
                            timeout=120,
                        )
                        file_id = up_res.get("file_id")
                        if not file_id:
                            raise RuntimeError("Le backend n'a pas renvoyé de file_id.")
                        # 2. Index
                        idx_res = api_json(
                            "POST",
                            f"{API_URL}/files/index",
                            params={"file_id": file_id, "model_name": st.session_state["model_name"]},
                            timeout=360,
                        )
                        st.sidebar.success(f"Indexé : {idx_res.get('indexed_chunks', 0)} fragments.")
                        if not idx_res.get("keyword_indexed", True):
                            st.sidebar.warning("Index vectoriel OK, index mots-clés indisponible.")
                    except Exception as e:
                        st.sidebar.error(f"Indexation impossible : {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📚 Documents Indexés**")

    try:
        files_registry = api_json("GET", f"{API_URL}/files", timeout=30)

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
                                try:
                                    api_json(
                                        "POST",
                                        f"{API_URL}/files/index",
                                        params={"file_id": fid, "model_name": st.session_state.get("model_name")},
                                        timeout=360,
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Indexation impossible : {e}")
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
                        try:
                            api_json("DELETE", f"{API_URL}/files/{fid}", timeout=60)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Suppression impossible : {e}")

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
                        try:
                            res = api_json("POST", f"{API_URL}/analyze", json=payload, timeout=360)
                            status.update(label="✅ Terminé", state="complete")
                            answer = res.get("report", {}).get("final_answer", "Erreur")
                            st.markdown(answer)
                            st.session_state["messages"].append({"role": "assistant", "content": answer})
                        except Exception as e:
                            status.update(label="❌ Échec", state="error")
                            st.error(f"Analyse impossible : {e}")
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
                    res = api_json(
                        "POST",
                        f"{API_URL}/ask-files",
                        params={"query": prompt, "model_name": st.session_state["model_name"]},
                        timeout=300,
                    )
                    if res.get("status") == "llm_error":
                        st.warning("Recherche réussie, mais génération LLM indisponible.")
                    elif res.get("status") == "retrieval_error":
                        st.warning("La recherche dans l'index documentaire a échoué.")
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
        files_registry = api_json("GET", f"{API_URL}/files", timeout=30)
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
                        res = api_json("POST", f"{API_URL}/compare-files", json=payload, timeout=300)
                        
                        if res.get("status") in ("success", "partial_success"):
                            st.success("Comparaison terminée !")
                            if res.get("status") == "partial_success":
                                st.warning("Résultat partiel : une comparaison de secours a été utilisée.")
                            
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
        files_registry = api_json("GET", f"{API_URL}/files", timeout=30)
        indexed_files = {fid: fdata for fid, fdata in files_registry.items() if fdata.get("indexed")}
    except Exception:
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
                    res = api_json("POST", f"{API_URL}/auto-audit", json=payload, timeout=420)
                    
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

elif app_mode == "Storage Hub":
    st.subheader("Storage Hub PostgreSQL")

    try:
        metrics = api_json("GET", f"{API_URL}/storage/metrics", timeout=30)
        col1, col2, col3 = st.columns(3)
        col1.metric("Backend", metrics.get("backend", "unknown"))
        col2.metric("Objets", metrics.get("objects", 0))
        col3.metric("Taille", f"{metrics.get('total_size_bytes', 0)} bytes")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Types")
            st.json(metrics.get("by_type", {}))
        with c2:
            st.markdown("### Statuts")
            st.json(metrics.get("by_status", {}))

        objects_payload = api_json("GET", f"{API_URL}/storage/objects", timeout=30)
        objects = objects_payload.get("objects", [])
        st.markdown("### Objets stockes")
        if objects:
            table_rows = [{
                "id": item.get("external_id"),
                "type": item.get("object_type"),
                "title": item.get("title"),
                "status": item.get("status"),
                "size": item.get("size_bytes"),
                "updated_at": item.get("updated_at"),
            } for item in objects]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        else:
            st.info("Aucun objet dans le Storage Hub.")

        events_payload = api_json("GET", f"{API_URL}/storage/events", params={"limit": 30}, timeout=30)
        events = events_payload.get("events", [])
        st.markdown("### Evenements recents")
        if events:
            event_rows = [{
                "event": event.get("event_type"),
                "object": event.get("external_id"),
                "actor": event.get("actor"),
                "message": event.get("message"),
                "created_at": event.get("created_at"),
            } for event in events]
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True)
        else:
            st.caption("Aucun evenement enregistre.")
    except Exception as e:
        st.error(f"Storage Hub indisponible : {e}")

elif app_mode == "Présentation Vocale":
    st.subheader("Agent de Présentation Vocale")

    try:
        files_registry = api_json("GET", f"{API_URL}/files", timeout=30)
    except Exception:
        files_registry = {}
    indexed_files = {fid: fdata for fid, fdata in files_registry.items() if fdata.get("indexed")}

    with st.form("voice_presentation_form"):
        topic = st.text_input("Sujet", value="Présente les principaux résultats disponibles")
        source_kind = st.selectbox("Source", ["auto", "dataset", "files", "custom"], index=0)
        selected_files = st.multiselect(
            "Documents",
            options=list(indexed_files.keys()),
            format_func=lambda fid: indexed_files[fid]["filename"],
        )
        language = st.selectbox("Langue vocale", ["fr-FR", "en-US", "es-ES"], index=0)
        tone = st.selectbox("Ton", ["professional", "executive", "teacher", "pitch"], index=0)
        duration = st.slider("Duree cible", min_value=1, max_value=12, value=4)
        user_context = st.text_area("Contexte libre", height=100)
        submitted = st.form_submit_button("Generer la presentation", type="primary")

    if submitted:
        payload = {
            "topic": topic,
            "source_kind": source_kind,
            "session_id": st.session_state.get("session_id"),
            "file_ids": selected_files,
            "user_context": user_context,
            "language": language,
            "tone": tone,
            "duration_minutes": duration,
            "model_name": st.session_state.get("model_name"),
        }
        with st.spinner("Generation de la presentation vocale..."):
            try:
                st.session_state["voice_presentation"] = api_json(
                    "POST",
                    f"{API_URL}/presentation/generate",
                    json=payload,
                    timeout=300,
                )
            except Exception as e:
                st.error(f"Generation impossible : {e}")

    presentation = st.session_state.get("voice_presentation")
    if presentation:
        st.markdown(f"### {presentation.get('title', 'Presentation')}")
        if presentation.get("executive_summary"):
            st.info(presentation["executive_summary"])
        if presentation.get("metadata", {}).get("fallback_used"):
            st.warning("Presentation generee en mode degrade.")

        render_voice_player(
            presentation.get("speech_text", ""),
            language=presentation.get("voice_profile", {}).get("language", "fr-FR"),
            rate=presentation.get("voice_profile", {}).get("rate", 1.0),
            pitch=presentation.get("voice_profile", {}).get("pitch", 1.0),
        )

        st.markdown("### Script")
        for index, segment in enumerate(presentation.get("segments", []), start=1):
            with st.expander(f"{index}. {segment.get('title', 'Segment')}", expanded=index == 1):
                st.write(segment.get("narration", ""))
                refs = segment.get("source_refs", [])
                if refs:
                    st.caption("Sources: " + ", ".join(refs))

        if presentation.get("limitations"):
            with st.expander("Limitations"):
                for limitation in presentation["limitations"]:
                    st.write(f"- {limitation}")
