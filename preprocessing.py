import os
import requests
from typing import List
from dotenv import dotenv_values

env_file = '.env' # for interactive sessions change to the correct path
config = dotenv_values(env_file)
for env_var in ['RAW_DATA_FOLDER','PREPROCESSED_DATA_FOLDER']:
    assert env_var in config, f'Could not find variable {env_var} in .env file: {env_file}'

def find_txt_files(root_dir: str) -> List[str]:
    txt_files = []
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

def process_text_files(input_folder: str, output_folder: str) -> None:
    assert os.path.exists(output_folder), f"Output folder {output_folder} does not exist."
        
    txt_files = find_txt_files(input_folder)
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        #text = clean_text_with_ollama(text) # Uncomment this line to use the Ollama API for text cleaning
        text_chunks = chunk_text(text)
        
        # Create output filename based on input filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Write each chunk to a separate file
        for i, chunk in enumerate(text_chunks):
            chunk_filename = f"{base_name}_chunk_{i}.txt"
            chunk_path = os.path.join(output_folder, chunk_filename)
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(chunk)



if __name__ == "__main__":
    input_folder = config['RAW_DATA_FOLDER']
    output_folder = config['PROCESSED_DATA_FOLDER']
    process_text_files(input_folder, output_folder)

