import pandas as pd

def get_smart_metadata(df: pd.DataFrame) -> dict:
    """Extrait l'ADN du dataset et nettoie les valeurs non-JSON (NaN)"""
    
    # Calcul des stats et remplacement des NaN par None pour la compatibilité JSON
    stats = df.describe(include='all').replace({float('nan'): None}).to_dict()
    
    # Échantillon nettoyé aussi
    sample = df.head(3).replace({float('nan'): None}).to_dict(orient="records")
    
    return {
        "columns": df.columns.tolist(),
        "types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample": sample,
        "stats": stats
    }
