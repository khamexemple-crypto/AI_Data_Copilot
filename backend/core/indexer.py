import pandas as pd
import os
import json
import logging
from backend.core.embeddings import get_embeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = ".chroma_db"
logger = logging.getLogger(__name__)

class DataIndexer:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=f"session_{session_id}"
        )

    def index_dataframe(self, df: pd.DataFrame, metadata: dict):
        """
        Découpe et indexe les métadonnées et un échantillon du DataFrame.
        """
        print(f"📦 [Indexer] Indexation des données pour la session {self.session_id}...")
        
        chunks = []
        ids = []
        metadatas = []

        # 1. Indexer la structure globale
        schema_info = {
            "type": "schema",
            "columns": metadata["columns"],
            "shape": metadata["shape"],
            "summary": f"Dataset avec {metadata['shape'][0]} lignes et {metadata['shape'][1]} colonnes."
        }
        chunks.append(json.dumps(schema_info))
        ids.append("schema_info")
        metadatas.append({"type": "schema"})

        # 2. Indexer chaque colonne individuellement
        for col in metadata["columns"]:
            col_info = {
                "type": "column",
                "name": col,
                "data_type": str(df[col].dtype),
                "null_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
            }
            chunks.append(f"Colonne '{col}': Type {col_info['data_type']}, {col_info['null_values']} valeurs nulles, {col_info['unique_values']} valeurs uniques.")
            ids.append(f"col_{col}")
            metadatas.append({"type": "column", "column_name": col})

        # 3. Indexer un échantillon structuré (ex: 50 premières lignes par blocs)
        sample_size = min(50, len(df))
        sample_df = df.head(sample_size)
        
        for i, row in sample_df.iterrows():
            row_text = f"Ligne {i}: " + ", ".join([f"{c}={v}" for c, v in row.items()])
            chunks.append(row_text)
            ids.append(f"row_{i}")
            metadatas.append({"type": "sample_row", "row_index": i})

        # Ajout à la collection Chroma via LangChain wrapper
        self.vectorstore.add_texts(
            texts=chunks,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ [Indexer] {len(chunks)} fragments indexés.")

def index_session_data(session_id: str, df: pd.DataFrame, metadata: dict):
    try:
        indexer = DataIndexer(session_id)
        indexer.index_dataframe(df, metadata)
        return True
    except Exception as e:
        logger.warning("index_session_data failed for session %s: %s", session_id, e)
        print(f"⚠️ [Indexer] Indexation ignorée pour la session {session_id}: {e}")
        return False
