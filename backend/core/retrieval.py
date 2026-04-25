from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from flashrank import Ranker, RerankRequest
from backend.core.embeddings import get_embeddings
import os

CHROMA_PATH = ".chroma_db"

class HybridRetriever:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.embeddings = get_embeddings()
        
        # Vector Store (Chroma)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=f"session_{session_id}"
        )
        
    def get_context(self, query: str, top_k: int = 5):
        """
        Récupère le contexte le plus pertinent via recherche hybride et reranking.
        """
        print(f"🔍 [Retriever] Recherche hybride pour: '{query}'")
        
        # 1. Recherche Vectorielle (Sémantique)
        vector_results = self.vectorstore.similarity_search(query, k=top_k*2)
        
        # 2. Recherche BM25 (Mots-clés)
        # Note: Pour BM25, on a besoin de tous les documents de la collection actuelle.
        all_docs = self.vectorstore.get()
        documents = all_docs['documents']
        if not documents:
            return ""
            
        bm25_retriever = BM25Retriever.from_texts(documents)
        bm25_retriever.k = top_k * 2
        bm25_results = bm25_retriever.get_relevant_documents(query)
        
        # 3. Fusion simple (Combine unique results)
        combined_docs = []
        seen = set()
        for doc in vector_results + bm25_results:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if content not in seen:
                combined_docs.append(content)
                seen.add(content)
                
        # 4. Reranking avec Flashrank (optionnel mais recommandé)
        # On simule un format compatible si Flashrank est utilisé
        try:
            ranker = Ranker()
            passages = [{"id": i, "text": doc} for i, doc in enumerate(combined_docs)]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)
            
            # On prend les top_k après reranking
            final_context = [r['text'] for r in results[:top_k]]
        except Exception as e:
            print(f"⚠️ [Retriever] Erreur Reranking ({e}), fallback sur fusion simple.")
            final_context = combined_docs[:top_k]
            
        return "\n---\n".join(final_context)

def retrieve_hybrid_context(session_id: str, query: str) -> str:
    retriever = HybridRetriever(session_id)
    return retriever.get_context(query)
