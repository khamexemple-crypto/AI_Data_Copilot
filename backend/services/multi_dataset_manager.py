import pandas as pd
from typing import Dict, List, Any

def load_multiple_datasets(files) -> Dict[str, pd.DataFrame]:
    """
    Loads a list of uploaded Streamlit files (or file paths) into pandas DataFrames.
    Returns a dict mapping filenames to their corresponding DataFrames.
    """
    datasets = {}
    for file in files:
        # Handle Streamlit UploadedFile objects which have 'name' attribute
        filename = getattr(file, "name", str(file))
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file)
            else:
                continue # Skip unsupported formats
            
            datasets[filename] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return datasets

def compare_datasets(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compares the schema (columns, types, row counts) of multiple datasets.
    """
    comparison = {}
    for name, df in datasets.items():
        comparison[name] = {
            "rows": len(df),
            "columns": len(df.columns),
            # Convert dtype objects to string for JSON serialization
            "schema": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        }
    return comparison

def suggest_merge_keys(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Analyzes common columns across all combinations of datasets to suggest merge keys.
    """
    if len(datasets) < 2:
        return {"error": "Need at least 2 datasets to suggest merges."}
    
    suggestions = {}
    names = list(datasets.keys())
    
    # Compare all pairs (combinations of 2)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]
            cols_a = set(datasets[name_a].columns)
            cols_b = set(datasets[name_b].columns)
            
            common_cols = list(cols_a.intersection(cols_b))
            
            pair_key = f"{name_a} <-> {name_b}"
            if common_cols:
                suggestions[pair_key] = {
                    "can_merge": True,
                    "suggested_keys": common_cols,
                    "message": f"Can be merged on: {', '.join(common_cols)}"
                }
            else:
                suggestions[pair_key] = {
                    "can_merge": False,
                    "suggested_keys": [],
                    "message": "No common columns found for a direct merge."
                }
                
    return suggestions
