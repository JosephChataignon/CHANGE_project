import os
import requests
from typing import List

def find_txt_files(root_dirs: List[str]) -> List[str]:
    txt_files = []
    for root_dir in root_dirs:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
    return txt_files

def clean_text_with_ollama(text: str, model: str = "llama2") -> str:
    url = "http://localhost:11434/api/generate"
    prompt = f"Clean and format the following text while preserving meaning: {text}"

    response = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama API error: {response.status_code}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

def process_text_files(folder_paths: List[str]) -> List[List[str]]:
    all_chunks = []
    txt_files = find_txt_files(folder_paths)

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned_text = clean_text_with_ollama(text)
        text_chunks = chunk_text(cleaned_text)
        all_chunks.append(text_chunks)

    return all_chunks