import pandas as pd
import requests
import json
import os
from any_llm import completion


def load_test_batterie(file_path='Testbatterie_FRAG_Rel&Val.xlsx'):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Verify that all expected columns are present
    expected_columns = ['Frage_ID', 'Handlungsfeld', 'Komplexitätsstufe', 'Aspekt', 'Frage']
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}. Available columns: {list(df.columns)}")

    return df

def search_frag_documents(query, base_url="http://change.dh.unibe.ch", n_results=5):
    api_url = f"{base_url}/search_documents/"
    data = {
        'query':query,
        'number_results':5
    }
    try:
        response = requests.post(
            api_url, 
            data=data
        )

        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Parse and return the JSON response
        return response.json()

    except requests.exceptions.ConnectionError:
        raise requests.RequestException(f"Failed to connect to {api_url}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from API")

def query_frag_api(prompt, base_url="http://change.dh.unibe.ch", n_results=5):
    api_url = f"{base_url}/api/chat/"
    payload = {
        "query": prompt.strip(),
        "n_results": n_results
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        # Send POST request to the API
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=3600 # set a very long timeout because requests can be quite long to process
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Parse and return the JSON response
        return response.json()
        
    except requests.exceptions.ConnectionError:
        raise requests.RequestException(f"Failed to connect to {api_url}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from API")



def query_anyllm(prompt, llm):
    
    if llm == "gpt5":
        anyllm_config = {'provider': 'openai', 'model': 'gpt-5', 'api_key': os.getenv('OPENAI_API_KEY')}
    elif llm == "deepseekR1":
        # Moved to query_ollama until bug is fixed in any-llm
        # anyllm_config = {'provider': 'ollama', 'model': 'deepseek-1.5', 'api_base': 'http://130.92.59.240:11434'} # TODO: check syntax
        return query_ollama(prompt, model='deepseek-r1:70b')
    
    response = completion(
        provider=anyllm_config['provider'], 
        model=anyllm_config['model'],
        api_key=anyllm_config.get('api_key', None),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def query_ollama(prompt, model):
    '''Need to add this because of a bug in any-llm Ollama provider.
    I submitted a pull request, hopefully it gets fixed soon and I can 
    run everything through any-llm and remove this function.
    '''
    response = requests.post(f'{os.getenv('OLLAMA_HOST')}/api/chat', 
        json={
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False
        })
    response_text = response.json()['message']['content']
    return response_text
    
