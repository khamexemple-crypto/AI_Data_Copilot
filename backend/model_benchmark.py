"""
Utilitaire de benchmarking pour les modèles Ollama locaux.
Permet d'évaluer la rapidité et la fiabilité de génération JSON de différents modèles.
"""

import time
import requests
from backend.core.config import settings
from backend.llm import call_llm, safe_json_parse

BENCHMARK_PROMPT = """You are Analyst Agent. Extract insights.
Return JSON ONLY. No markdown.
{
  "insights": ["insight 1"],
  "confidence": 0.9
}

User query: Analyze sales for Q3.
Dataset info:
Shape: (100, 3)
Cols: Date, Sales, Region"""

def benchmark_model(model_name: str) -> dict:
    print(f"\n🚀 Benchmarking model: {model_name}")
    start_time = time.perf_counter()
    
    try:
        raw_response = call_llm(prompt=BENCHMARK_PROMPT, timeout=60, model_name=model_name)
        elapsed = round(time.perf_counter() - start_time, 2)
        
        parsed = safe_json_parse(raw_response)
        
        is_valid = parsed is not None and "insights" in parsed and "confidence" in parsed
        
        print(f"⏱️  Time: {elapsed}s")
        print(f"✅ Valid JSON: {is_valid}")
        
        return {
            "model": model_name,
            "status": "success",
            "time_sec": elapsed,
            "valid_json": is_valid,
            "raw_preview": raw_response[:100].replace("\n", " ") + "..."
        }
        
    except Exception as e:
        elapsed = round(time.perf_counter() - start_time, 2)
        print(f"❌ Error: {e}")
        return {
            "model": model_name,
            "status": "error",
            "time_sec": elapsed,
            "error_msg": str(e)
        }

def benchmark_all_models():
    print("=========================================")
    print("📊 STARTING LOCAL LLM BENCHMARK")
    print("=========================================")
    
    results = []
    for model in settings.AVAILABLE_MODELS:
        res = benchmark_model(model)
        results.append(res)
        
    print("\n=========================================")
    print("🏆 BENCHMARK RESULTS SUMMARY")
    print("=========================================")
    print(f"{'MODEL':<25} | {'STATUS':<10} | {'TIME (s)':<10} | {'JSON OK':<10}")
    print("-" * 65)
    for r in results:
        status = r['status']
        time_s = r.get('time_sec', '-')
        json_ok = "YES" if r.get('valid_json') else "NO" if status == "success" else "-"
        print(f"{r['model']:<25} | {status:<10} | {time_s:<10} | {json_ok:<10}")

if __name__ == "__main__":
    benchmark_all_models()
