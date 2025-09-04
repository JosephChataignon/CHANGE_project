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
            timeout=60
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Parse and return the JSON response
        return response.json()
        
    except requests.exceptions.ConnectionError:
        raise requests.RequestException(f"Failed to connect to {api_url}")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400:
            error_msg = response.json().get('error', 'Bad request')
            raise ValueError(f"API error: {error_msg}")
        elif response.status_code == 500:
            error_msg = response.json().get('error', 'Internal server error')
            raise requests.RequestException(f"Server error: {error_msg}")
        else:
            raise requests.RequestException(f"HTTP {response.status_code}: {str(e)}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from API")



def query_anyllm(prompt, llm):
    
    if llm == "gpt5":
        anyllm_config = {'provider': 'openai', 'model': 'gpt-5', 'api_key': os.getenv('OPENAI_API_KEY')}
    elif llm == "deepseek":
        anyllm_config = {'provider': 'ollama', 'model': 'deepseek-1.5'} # TODO: check syntax
    
    response = completion(
        provider=anyllm_config['provider'], 
        model=anyllm_config['model'],
        api_key=anyllm_config.get('api_key', None),
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
