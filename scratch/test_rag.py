import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
from backend.core.indexer import index_session_data
from backend.core.retrieval import retrieve_hybrid_context

# Mock data
data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["Paris", "London", "Berlin"]
}
df = pd.DataFrame(data)
metadata = {
    "columns": ["name", "age", "city"],
    "shape": [3, 3],
    "sample": data
}
session_id = "test_session_rag"

print("--- Testing Indexing ---")
index_session_data(session_id, df, metadata)

print("\n--- Testing Retrieval ---")
query = "Quelles sont les villes disponibles ?"
context = retrieve_hybrid_context(session_id, query)
print(f"Query: {query}")
print(f"Context Found:\n{context}")

query_age = "Quel est l'âge moyen ?"
context_age = retrieve_hybrid_context(session_id, query_age)
print(f"\nQuery: {query_age}")
print(f"Context Found:\n{context_age}")
