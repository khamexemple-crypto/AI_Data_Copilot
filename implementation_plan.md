# Goal: AI Data Copilot Intelligent File Knowledge Base

## 1. FEATURE EXPLANATION

This upgrade transforms the platform from a strict tabular dataset analyzer into a **hybrid knowledge base**. 
Users can now upload unstructured documents (PDF, TXT, DOCX) and structured files (JSON, CSV, XLSX). The system will persist these files, index their contents into a local Vector Store (ChromaDB), and allow semantic querying using a new RAG (Retrieval-Augmented Generation) pipeline. This enables users to ask cross-domain questions like: *"How do our dataset sales compare to the goals outlined in the Q3 Business Plan PDF?"*

## 2. UPDATED ARCHITECTURE

```text
User 
 ├──> Uploads Dataset ──> Pandas / CSV
 └──> Uploads Document ──> File Storage ──> Document Loader ──> Text Chunker 
                                                                    │
                                                           Embedding Generator (Sentence Transformers)
                                                                    │
                                                               Vector Store (ChromaDB)
                                                                    │
User ──> Asks Question ──> Planner Agent ───(routes)───> Retriever (Semantic Search)
                                 │                                  │
                          (Dataset Route)                     (File Route)
                                 │                                  │
                           Analyst Agent                        RAG Agent
                                 │                                  │
                                 └───> Reporter Agent <─────────────┘
                                           │
                                     Final Answer (with Sources)
```

## 3. FINAL FILE STRUCTURE

**New Directories & Files:**
```text
backend/
├── storage/
│   ├── __init__.py
│   ├── file_manager.py     # Saves/deletes files from disk
│   └── file_registry.py    # In-memory or SQLite index of stored files
├── rag/
│   ├── __init__.py
│   ├── document_loader.py  # PDF, TXT, DOCX parsers
│   ├── chunker.py          # Recursive text splitting
│   ├── vector_store.py     # ChromaDB wrapper
│   └── retriever.py        # Semantic search logic
└── agents/
    └── rag_agent.py        # LLM agent to synthesize answers from chunks
```
**Modified Files:**
- `backend/main.py` or `backend/api/routes.py`: Add `/files/upload`, `/files`, `/files/index`, `/ask-files`.
- `backend/agents/planner.py`: Update routing logic to detect file vs dataset queries.
- `frontend/app.py`: Add "📁 File Knowledge Base" sidebar and rendering logic for sources.

## 4. BACKEND IMPLEMENTATION PLAN

- **Storage**: Save uploaded files in a local `data/uploads` folder. Keep a `files.json` registry mapping `file_id` to `filename` and `type`.
- **Document Loading**: Use `pypdf` for PDFs, `python-docx` for DOCX, and plain Python `open()` for TXT. For CSV/JSON, dump as formatted string.
- **Chunking**: Use a simple character-based chunker with overlap (e.g., 1000 chars, 200 overlap).
- **Vector Store & Embeddings**: Use `chromadb` with its default `all-MiniLM-L6-v2` sentence-transformer. It runs locally and requires zero API configuration.
- **RAG Pipeline**: 
  1. User asks question.
  2. Query is embedded and sent to ChromaDB.
  3. Top 3-5 chunks are retrieved.
  4. Chunks are passed in the prompt to the `RAG Agent` using the locally selected Ollama model.
- **API Endpoints**: FastAPIRouter endpoints bridging the storage and RAG modules.

## 5. FRONTEND IMPLEMENTATION PLAN

- **Sidebar**: Add an expander or section for `📁 File Knowledge Base`. Include a file uploader `st.file_uploader(accept_multiple_files=True)` and an "Index Files" button.
- **Main Chat**: Modify the chat interface. When an answer is received from the RAG pipeline, check for a `"sources"` key. If present, render them in an `st.expander("📚 Sources retrieved")` showing the filename and excerpt.

---

## 6. FULL CODE FOR CORE FILES

### `backend/storage/file_manager.py`
```python
import os
import uuid
import shutil
from fastapi import UploadFile

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file: UploadFile) -> str:
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'txt'
    safe_filename = f"{file_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_id, file_path
    
def get_file_path(file_id: str, filename: str) -> str:
    ext = filename.split('.')[-1] if '.' in filename else 'txt'
    return os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

def delete_file(file_id: str, filename: str):
    path = get_file_path(file_id, filename)
    if os.path.exists(path):
        os.remove(path)
```

### `backend/storage/file_registry.py`
```python
import json
import os

REGISTRY_PATH = "data/file_registry.json"

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(data):
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=4)

def register_file(file_id: str, filename: str, file_type: str):
    registry = load_registry()
    registry[file_id] = {"filename": filename, "type": file_type, "indexed": False}
    save_registry(registry)

def mark_indexed(file_id: str):
    registry = load_registry()
    if file_id in registry:
        registry[file_id]["indexed"] = True
        save_registry(registry)

def get_all_files():
    return load_registry()
```

### `backend/rag/document_loader.py`
```python
import pandas as pd
import json

def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    text = ""
    
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == 'pdf':
        import pypdf
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == 'docx':
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif ext in ['csv', 'xlsx']:
        df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)
        text = f"Dataset Columns: {list(df.columns)}\nSample Data:\n{df.head(5).to_csv(index=False)}"
    elif ext == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = json.dumps(data, indent=2)
            
    return text
```

### `backend/rag/chunker.py`
```python
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += (chunk_size - chunk_overlap)
        
    return chunks
```

### `backend/rag/vector_store.py`
```python
import chromadb
from chromadb.utils import embedding_functions

# Utilise le modèle local par défaut all-MiniLM-L6-v2 de sentence-transformers
client = chromadb.PersistentClient(path="./data/chroma_db")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(name="copilot_docs", embedding_function=emb_fn)

def add_chunks_to_store(file_id: str, filename: str, chunks: list):
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"file_id": file_id, "filename": filename, "chunk_id": str(i)} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

def search_store(query: str, n_results: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    retrieved = []
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        for d, m in zip(docs, metas):
            retrieved.append({
                "excerpt": d,
                "file_id": m.get("file_id"),
                "filename": m.get("filename"),
                "chunk_id": m.get("chunk_id")
            })
    return retrieved
```

### `backend/agents/rag_agent.py`
```python
import json
from backend.llm import call_llm, safe_json_parse

RAG_SYSTEM_PROMPT = """You are a precise RAG Agent.
Answer the user's question using ONLY the provided retrieved context.
If the context is insufficient, state clearly that you cannot answer based on the files.
Do not hallucinate.

Return ONLY a valid JSON object matching this schema:
{
  "answer": "Your detailed answer",
  "confidence": 0.9,
  "limitations": ["Information missing about X"],
  "used_sources": ["filename1.pdf", "filename2.csv"]
}"""

def run_rag_agent(question: str, retrieved_chunks: list, model_name: str = None) -> dict:
    context_text = "\n\n---\n\n".join([f"Source: {c['filename']} (Chunk {c['chunk_id']})\n{c['excerpt']}" for c in retrieved_chunks])
    
    user_prompt = f"Question: {question}\n\nRetrieved Context:\n{context_text}"
    
    raw_response = call_llm(prompt=user_prompt, system=RAG_SYSTEM_PROMPT, timeout=60, model_name=model_name)
    parsed = safe_json_parse(raw_response)
    
    if not parsed or "answer" not in parsed:
        return {"answer": "Error generating answer from context.", "confidence": 0.0, "limitations": ["LLM parse error"]}
        
    # Inject exact sources metadata into output
    parsed["sources"] = retrieved_chunks
    return parsed
```

### `backend/api/routes.py` (Endpoint additions)
```python
from fastapi import APIRouter, UploadFile, File
from backend.storage import file_manager, file_registry
from backend.rag import document_loader, chunker, vector_store
from backend.agents.rag_agent import run_rag_agent

# ... existing routes ...

@router.post("/files/upload")
def upload_document(file: UploadFile = File(...)):
    file_id, _ = file_manager.save_uploaded_file(file)
    ext = file.filename.split('.')[-1]
    file_registry.register_file(file_id, file.filename, ext)
    return {"status": "success", "file_id": file_id, "filename": file.filename}

@router.get("/files")
def list_files():
    return file_registry.get_all_files()

@router.post("/files/index")
def index_file(file_id: str):
    registry = file_registry.get_all_files()
    if file_id not in registry: return {"status": "error", "message": "File not found"}
    
    filename = registry[file_id]["filename"]
    path = file_manager.get_file_path(file_id, filename)
    
    text = document_loader.extract_text(path, filename)
    chunks = chunker.chunk_text(text)
    vector_store.add_chunks_to_store(file_id, filename, chunks)
    
    file_registry.mark_indexed(file_id)
    return {"status": "success", "indexed_chunks": len(chunks)}

@router.post("/ask-files")
def ask_files(query: str, model_name: str = None):
    chunks = vector_store.search_store(query)
    if not chunks:
        return {"status": "success", "answer": "No relevant files found.", "sources": []}
        
    answer_dict = run_rag_agent(query, chunks, model_name)
    answer_dict["status"] = "success"
    return answer_dict
```

### `frontend/app.py` (Modifications)
```python
# Sidebar
st.sidebar.header("📁 File Knowledge Base")
kb_files = st.sidebar.file_uploader("Upload Documents (PDF, DOCX, TXT...)", accept_multiple_files=True)
if kb_files:
    for f in kb_files:
        if st.sidebar.button(f"Upload & Index {f.name}"):
            with st.spinner("Processing..."):
                res = requests.post(f"{API_URL}/files/upload", files={"file": (f.name, f.getvalue())}).json()
                idx_res = requests.post(f"{API_URL}/files/index?file_id={res['file_id']}").json()
                st.sidebar.success(f"Indexed {idx_res.get('indexed_chunks', 0)} chunks!")

# Dans la gestion du chat /analyze
elif mode == "files":
    res = requests.post(f"{API_URL}/ask-files", params={"query": prompt, "model_name": selected_model}).json()
    st.markdown(res.get("answer", ""))
    
    if res.get("sources"):
        with st.expander("📚 Sources Retrieved"):
            for src in res["sources"]:
                st.markdown(f"**{src['filename']}**")
                st.caption(f"_{src['excerpt']}_")
```

---

## 7. EXAMPLE REQUESTS

**Upload & Index:**
```json
POST /api/files/upload (form-data: file=business_plan.pdf)
POST /api/files/index?file_id=abc-123
```

**Ask Question:**
```json
POST /api/ask-files?query=Quels sont les objectifs Q3 ?&model_name=llama3.2:3b
```

## 8. EXAMPLE RESPONSES

```json
{
  "status": "success",
  "answer": "D'après le Business Plan, les objectifs Q3 sont d'augmenter les ventes de 15% et d'ouvrir deux nouvelles franchises.",
  "confidence": 0.95,
  "limitations": ["Le document ne précise pas quelles villes sont ciblées pour les franchises."],
  "used_sources": ["business_plan.pdf"],
  "sources": [
    {
      "file_id": "abc-123",
      "filename": "business_plan.pdf",
      "chunk_id": "4",
      "excerpt": "...objectifs stratégiques pour Q3 incluent une hausse de 15% du CA et 2 franchises..."
    }
  ]
}
```

## 9. HOW TO TEST LOCALLY

1. Install required packages: `pip install chromadb pypdf python-docx sentence-transformers`.
2. Start the FastAPI backend.
3. Use the Streamlit sidebar to upload a sample PDF (e.g., a dummy business plan).
4. Click "Upload & Index". You should see a success message with the number of chunks.
5. In the chat, type a question specifically related to the PDF and trigger the RAG flow.
6. Verify that the answer cites the PDF and that the exact text chunks appear in the "Sources Retrieved" expander.

## 10. COMMON ERRORS TO AVOID

1. **Missing Dependencies**: `pypdf`, `python-docx`, and `chromadb` must be added to `requirements.txt`.
2. **Context Overflow**: If `chunk_size` is too large and you retrieve `n_results=10`, you will crash smaller local models like `phi3:mini`. Stick to 3-4 chunks of 1000 chars.
3. **Database Locks**: ChromaDB PersistentClient holds a lock. If you run multiple FastAPI workers (e.g. `uvicorn --workers 4`), it will crash. Stick to 1 worker locally.
4. **Hallucinations on missing context**: Ensure `RAG_SYSTEM_PROMPT` explicitly forbids answering if the context doesn't contain the info. Otherwise, the LLM will fall back on its pre-trained knowledge.
5. **Cross-Session Privacy**: `file_registry.json` is global. In a real multi-user environment, `session_id` must be tied to `file_id`. For this PFA demo, global storage is acceptable.
