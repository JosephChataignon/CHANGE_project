import os, sys, requests
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

def process_text_files(input_folder: str, output_folder: str) -> None:
    assert os.path.exists(output_folder), f"Output folder {output_folder} does not exist."

    txt_files = find_txt_files(input_folder)

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        cleaned_text = clean_text_with_ollama(text)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}_cleaned.txt"
        output_path = os.path.join(output_folder, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

if __name__ == "__main__":
    var = sys.argv[1]
    if var == "clean":
        # clean the data
        process_text_files(config['RAW_DATA_FOLDER'], config['PREPROCESSED_DATA_FOLDER'])
    elif var == "eval":
        # evaluate cleaning process
        # here we're going to compare the cleaned data to the original data, 
        # based on the sample folder.
        # TODO: 
        # - add sample folder path
        # - write the evaluation function
        # - change process_text_files to allow evaluate without writing 
        #   processed files but writing the eval result
        # - call process_text_files in eval mode
        pass
        
