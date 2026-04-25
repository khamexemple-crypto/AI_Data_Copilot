from backend.llm import call_llm, safe_json_parse
from backend.prompts import FILE_SUMMARY_PROMPT

def generate_file_intelligence(text: str, model_name: str = None) -> dict:
    # Limit text to the first 4000 characters to prevent context overflow and save time
    truncated_text = text[:4000]
    
    user_prompt = f"Document Text:\n{truncated_text}\n\nGenerate the JSON intelligence metadata."
    
    raw_response = call_llm(prompt=user_prompt, system=FILE_SUMMARY_PROMPT, timeout=45, model_name=model_name)
    parsed = safe_json_parse(raw_response)
    
    default_meta = {
        "summary": "No summary available.",
        "tags": ["document"],
        "key_topics": [],
        "suggested_questions": ["De quoi parle ce document ?"]
    }
    
    if parsed and "summary" in parsed:
        return parsed
    return default_meta
