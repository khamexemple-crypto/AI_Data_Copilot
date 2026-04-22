# Base de données temporaire en mémoire
# Les clés sont les session_id (String), les valeurs sont des dict avec {"dataframe": df, "metadata": meta}
session_storage = {}
