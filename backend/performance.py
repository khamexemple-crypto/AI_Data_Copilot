import time
from typing import Callable, Any, Tuple

def time_it(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Exécute une fonction et retourne le résultat ainsi que le temps écoulé en secondes.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = round(time.perf_counter() - start_time, 2)
    return result, elapsed

def compress_metadata(metadata: dict, sample_rows: list) -> str:
    """
    Compresses dataset context to reduce LLM token usage.
    """
    shape = metadata.get("shape", "Inconnu")
    columns = metadata.get("columns", [])
    
    # We only take the most crucial info
    compressed = f"Shape: {shape}\nCols: {', '.join(columns)}\n"
    
    # Add types if available
    if "dtypes" in metadata:
        compressed += f"Types: {metadata['dtypes']}\n"
        
    # Add missing values if available
    if "missing_values" in metadata:
        missing = {k: v for k, v in metadata["missing_values"].items() if v > 0}
        if missing:
            compressed += f"Missing: {missing}\n"

    # Add max 3 rows
    if sample_rows:
        compressed += f"Sample (3 rows):\n{sample_rows[:3]}\n"
        
    return compressed

