import pandas as pd
import json
import pypdf
import docx

def extract_text(file_path: str, filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    text = ""
    
    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == 'pdf':
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext in ['csv', 'xlsx']:
            df = pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)
            text = f"Dataset Columns: {list(df.columns)}\nSample Data:\n{df.head(5).to_csv(index=False)}"
        elif ext == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        text = ""
            
    return text
