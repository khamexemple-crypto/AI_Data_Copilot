import pandas as pd
import traceback

def execute_python_code(code: str, df: pd.DataFrame) -> dict:
    """
    Exécute le code généré par l'agent Coder dans un environnement restreint.
    Le DataFrame `df` y est injecté en lecture/écriture en mémoire.
    Les résultats doivent être bindés aux variables locales `result` ou `plot_json`.
    """
    local_env = {
        "df": df,
        "pd": pd,
        "result": None,
        "plot_json": None
    }
    
    try:
        exec(code, {}, local_env)
        return {
            "success": True,
            "result": local_env.get("result"),
            "plot_json": local_env.get("plot_json")
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "error": error_trace
        }
