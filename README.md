# 🤖 AI Data Copilot

**AI Data Copilot** est une plateforme autonome d'analyse de données propulsée par l'Intelligence Artificielle. Conçue pour simplifier le travail des data analysts, elle utilise une architecture multi-agents pour traiter, analyser et visualiser des datasets locaux en toute sécurité.

---

## 🎯 Problématique

Les professionnels de la donnée passent un temps considérable à nettoyer, explorer et générer des rapports initiaux sur de nouveaux datasets. Les solutions cloud existantes posent souvent des problèmes de confidentialité des données (envoi de fichiers sensibles sur des serveurs tiers) et manquent d'adaptabilité pour des tâches complexes nécessitant plusieurs étapes de réflexion.

## 💡 Solution

**AI Data Copilot** résout ce problème en proposant un assistant 100% local (privacy-first) capable de prendre en charge l'intégralité du cycle d'analyse exploratoire (EDA). Grâce à une architecture multi-agents orchestrée par FastAPI et Streamlit, le système planifie, analyse, critique et synthétise les données de manière autonome.

---

## ✨ Fonctionnalités Principales

- **Ingestion de Données** : Support natif des fichiers `.csv` et `.xlsx`.
- **Profilage Autonome** : Extraction intelligente des métadonnées (shape, types, valeurs manquantes) sans exposer les données brutes complètes au LLM.
- **Architecture Multi-Agents** :
  - *Planner Agent* : Comprend l'intention de l'utilisateur et élabore un plan.
  - *Analyst Agent* : Extrait les insights, anomalies et corrélations.
  - *Reviewer Agent* : Évalue la qualité de l'analyse et relève les limites.
  - *Reporter Agent* : Synthétise le travail en une réponse claire.
- **Modes d'Exécution** :
  - ⚡ *Mode Rapide* : Utilise des modèles légers (ex: Llama 3.2 3B) pour une exécution quasi-instantanée en une passe.
  - 🧠 *Mode Complet* : Utilise toute la chaîne multi-agents pour une réflexion profonde (ex: DeepSeek Coder V2).
- **Optimisation des Performances** : Cache en mémoire, compression des prompts, fallbacks de sécurité en cas d'indisponibilité du LLM.

---

## 🛠 Tech Stack

- **Backend** : Python 3.10+, FastAPI, SQLAlchemy
- **Frontend** : Streamlit
- **IA / LLM** : Ollama (Exécution locale de modèles open-source)
- **Data Science** : Pandas, Plotly

---

## 📂 Structure du Projet

```text
AI_Data_Copilot/
├── backend/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── api/routes.py           # Endpoints de l'application
│   ├── core/config.py          # Configuration & Modèles
│   ├── llm.py                  # Communication avec Ollama
│   ├── orchestrator.py         # Chef d'orchestre du pipeline
│   ├── agents/                 # Logique des agents (Planner, Analyst, etc.)
│   └── cache.py                # Système de cache intelligent
├── frontend/
│   └── app.py                  # Interface Utilisateur Streamlit
└── requirements.txt            # Dépendances Python
```

---

## 🚀 Installation & Démarrage

### 1. Prérequis
- Python 3.10 ou supérieur
- [Ollama](https://ollama.com/) installé sur votre machine.

### 2. Téléchargement des modèles LLM
Ouvrez un terminal et téléchargez les modèles recommandés :
```bash
ollama pull deepseek-coder-v2:lite
ollama pull llama3.2:3b
```

### 3. Installation des dépendances
```bash
# Création d'un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installation des librairies
pip install -r requirements.txt
```

### 4. Lancement de l'application
Vous devez lancer deux terminaux distincts :

**Terminal 1 : Backend (FastAPI)**
```bash
source venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 : Frontend (Streamlit)**
```bash
source venv/bin/activate
streamlit run frontend/app.py --server.port 8502
```

L'interface sera accessible sur `http://localhost:8502`.

---

## 🔌 API Endpoints

- `POST /api/upload` : Uploade le dataset et initie une session.
- `POST /api/profile` : Génère l'audit de santé autonome des données.
- `POST /api/analyze` : Déclenche le pipeline (Fast ou Deep) selon la requête utilisateur.
- `GET /api/models` : Liste les modèles Ollama configurés.
- `GET /api/benchmark-models` : Évalue la vitesse des modèles en direct.

---

## ⚠️ Limites Actuelles

- Les graphiques (Plotly) générés par le LLM dépendent fortement des capacités du modèle en matière d'écriture de code Python.
- L'historique contextuel complet n'est pas encore conservé entre chaque question du chat, chaque requête est traitée en "one-shot" avec le dataset en contexte.
- L'application est optimisée pour des ordinateurs dotés de puces Apple Silicon (M1/M2/M3) ou de cartes graphiques dédiées pour faire tourner Ollama fluidement.

---

## 🔮 Améliorations Futures

- **RAG (Retrieval-Augmented Generation)** : Intégration d'une base vectorielle (ChromaDB) pour discuter avec des bases de données immenses sans saturer le contexte du LLM.
- **Exécution de Code Autonome** : Permettre au modèle d'exécuter du code Python dans un environnement sandboxé sécurisé pour valider ses propres hypothèses.
- **Export PDF** : Ajout d'un bouton pour télécharger le rapport d'audit finalisé au format PDF.

---

## 👥 Auteur
Développé dans le cadre du **Projet de Fin d'Année (PFA)**.
*Propulsé par la puissance de l'open source.*
