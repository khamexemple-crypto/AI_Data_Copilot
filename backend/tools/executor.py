import pandas as pd
import traceback
import ast
import concurrent.futures

def check_code_safety(code: str):
    """
    Vérifie l'AST pour bloquer les imports dangereux.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names = [node.module.split('.')[0]]
            
            for mod in module_names:
                if mod in ['os', 'sys', 'subprocess', 'shutil', 'socket', 'pathlib', 'builtins']:
                    raise ValueError(f"Dangerous import blocked: {mod}")

def _run_exec(code: str, local_env: dict):
    # Restrict available globals: no builtins like __import__ or open
    safe_globals = {
        "__builtins__": {
            "print": print, "range": range, "len": len, "int": int, "float": float,
            "str": str, "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "isinstance": isinstance, "Exception": Exception,
            "ValueError": ValueError, "TypeError": TypeError
        }
    }
    exec(code, safe_globals, local_env)
    return local_env

def execute_python_code(code: str, df: pd.DataFrame, timeout: int = 15) -> dict:
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
        check_code_safety(code)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_exec, code, local_env)
            updated_env = future.result(timeout=timeout)
            
        return {
            "success": True,
            "result": updated_env.get("result"),
            "plot_json": updated_env.get("plot_json")
        }
    except concurrent.futures.TimeoutError:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout} seconds."
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "success": False,
            "error": error_trace
        }
