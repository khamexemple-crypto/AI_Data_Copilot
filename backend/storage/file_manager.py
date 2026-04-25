import os
import uuid
import shutil
from fastapi import UploadFile

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(file: UploadFile) -> str:
    file_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'txt'
    safe_filename = f"{file_id}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_id, file_path
    
def get_file_path(file_id: str, filename: str) -> str:
    ext = filename.split('.')[-1] if '.' in filename else 'txt'
    return os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

def delete_file(file_id: str, filename: str):
    path = get_file_path(file_id, filename)
    if os.path.exists(path):
        os.remove(path)
