
import sys, logging, csv
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
from any_llm import AnyLLM

from data import get_CHANGE_data_random_samples
from logging_utils import setup_logging



def load_samples(sample_size: int):
    data_storage = config['DATA_STORAGE']
    assert data_storage is not None, "DATA_STORAGE environment variable not set"
    return get_CHANGE_data_random_samples(sample_size, data_storage)

def _initialize_llm():
    provider = config.get('llm_provider')
    api_key = config.get('llm_api_key')
    api_base = config.get('llm_api_base')
    api_base = None if api_base == '' else api_base
    
    llm = AnyLLM.create(provider, api_key=api_key, api_base=api_base)
    return llm

def _call_to_LLM(llm, prompt):
    model_name = config.get('llm_model')
    response = llm.completion(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a RAG system."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
    )
    return response.choices[0].message.content

def filter_sample(llm, sample):
    response = _call_to_LLM(llm, f"""Is the following text a paragraph of text 
    for educational BLABLABLA TODO
    purposes? Answer with 'yes' or 'no'. <TEXT_START> {sample} <TEXT_END>""")
    return response.strip().lower() == 'yes'

def generate_query(llm, sample):
    response = _call_to_LLM(llm, f"""Generate a question that could be answered
    by the following text: <TEXT_START> {sample} <TEXT_END>. The question should be placed
    between <QUERY_START> and <QUERY_END> tags.""")
    return response.split('<QUERY_START>')[1].split('<QUERY_END>')[0].strip()

def generate_synthetic_data(size):
    # 1. Load samples and initialize LLM
    samples = load_samples(size)
    llm = _initialize_llm()
    # 2. Filter samples
    filtered_samples = [sample for sample in samples if filter_sample(llm, sample)]
    # 3. Generate queries
    data = []
    for sample in filtered_samples:
        query = generate_query(llm, sample)
        if query and len(query) > 10:
            data.append((query, sample))

    return data



if __name__ == "__main__":
    ## Load argument
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    ## Load environment variables
    env_file = '.env' # for interactive sessions change to the correct path
    config  = dotenv_values(env_file)
    for env_var in ['LOGS_FOLDER','llm_provider', 'llm_api_key','llm_api_base','llm_model','DATA_STORAGE']:
        assert env_var in config, f'Could not find variable {env_var} in .env file: {env_file}'

    ## Setup logging
    start_time = datetime.now()
    root_logger = logging.getLogger()
    setup_logging(config, root_logger)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(size)
    
    # Save to csv in (query, relevant_sentence) format
    writer = csv.writer(open("synthetic_data.csv", 'w', newline=''))
    writer.writerow(['query', 'relevant_sentence'])
    for query, sentence in synthetic_data:
        writer.writerow([query, sentence]) 
    writer.close()


