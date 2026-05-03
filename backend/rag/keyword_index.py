import os
import json
from rank_bm25 import BM25Okapi

# Simple persistent index
BM25_STORE_PATH = "data/bm25_store.json"

class KeywordIndex:
    def __init__(self):
        self.corpus = []
        self.metadatas = []
        self.bm25 = None
        self.load()

    def load(self):
        if os.path.exists(BM25_STORE_PATH):
            try:
                with open(BM25_STORE_PATH, 'r') as f:
                    data = json.load(f)
                    self.corpus = data.get("corpus", [])
                    self.metadatas = data.get("metadatas", [])
                    self._rebuild()
            except Exception as e:
                print(f"Error loading BM25 index: {e}")
                self.corpus = []
                self.metadatas = []
                self.bm25 = None

    def _rebuild(self):
        if self.corpus:
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def save(self):
        os.makedirs("data", exist_ok=True)
        with open(BM25_STORE_PATH, 'w') as f:
            json.dump({"corpus": self.corpus, "metadatas": self.metadatas}, f)

    def remove_file(self, file_id: str, save: bool = True):
        kept = [
            (doc, meta)
            for doc, meta in zip(self.corpus, self.metadatas)
            if meta.get("file_id") != file_id
        ]
        self.corpus = [doc for doc, _ in kept]
        self.metadatas = [meta for _, meta in kept]
        self._rebuild()
        if save:
            self.save()

    def add_chunks(self, chunks: list, metadatas: list):
        file_ids = {m.get("file_id") for m in metadatas if m.get("file_id")}
        for file_id in file_ids:
            self.remove_file(file_id, save=False)

        self.corpus.extend(chunks)
        self.metadatas.extend(metadatas)
        self._rebuild()
        self.save()

    def search(self, query: str, top_k: int = 10, file_ids: list[str] = None):
        if not self.bm25: return []
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Sort all indices
        all_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        
        results = []
        for idx in all_indices:
            # Filter by file_ids if provided
            if file_ids and self.metadatas[idx].get("file_id") not in file_ids:
                continue
                
            score = doc_scores[idx]
            if score > 0:
                results.append({
                    "text": self.corpus[idx],
                    "score": float(score),
                    **self.metadatas[idx]
                })
                
            if len(results) >= top_k:
                break
                
        return results

keyword_store = KeywordIndex()
