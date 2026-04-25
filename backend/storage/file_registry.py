import json
import os

REGISTRY_PATH = "data/file_registry.json"

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(data):
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=4)

def register_file(file_id: str, filename: str, file_type: str):
    registry = load_registry()
    registry[file_id] = {"filename": filename, "type": file_type, "indexed": False}
    save_registry(registry)

def update_file_metadata(file_id: str, metadata: dict):
    registry = load_registry()
    if file_id in registry:
        registry[file_id].update(metadata)
        save_registry(registry)

def mark_indexed(file_id: str):
    registry = load_registry()
    if file_id in registry:
        registry[file_id]["indexed"] = True
        save_registry(registry)

def get_all_files():
    return load_registry()
