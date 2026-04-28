import chromadb
from chromadb.utils import embedding_functions
import os

# Create directory if it doesn't exist
CHROMA_PATH = "./data/chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

# Utilise le modèle local par défaut all-MiniLM-L6-v2 de sentence-transformers
client = chromadb.PersistentClient(path=CHROMA_PATH)
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

def search_store(query: str, n_results: int = 10, file_ids: list[str] = None):
    where_clause = None
    if file_ids:
        if len(file_ids) == 1:
            where_clause = {"file_id": file_ids[0]}
        else:
            where_clause = {"file_id": {"$in": file_ids}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )
    
    retrieved = []
    if results['documents'] and len(results['documents']) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [0.5] * len(docs)
        
        for d, m, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": d,
                "file_id": m.get("file_id"),
                "filename": m.get("filename"),
                "chunk_id": m.get("chunk_id"),
                "distance": dist
            })
    return retrieved
