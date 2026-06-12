
import os, random, logging, csv
from datetime import datetime
from dotenv import load_dotenv, dotenv_values

from data import get_CHANGE_data_random_samples
from logging_utils import setup_logging



def load_samples(sample_size: int):
    data_storage = config['DATA_STORAGE']
    assert data_storage is not None, "DATA_STORAGE environment variable not set"
    return get_CHANGE_data_random_samples(sample_size, data_storage)

def _call_to_LLM(prompt):
    pass

def filter_sample(sample):
    response = _call_to_LLM(f"""Is the following text a paragraph of text 
    for educational BLABLABLA TODO
    purposes? Answer with 'yes' or 'no'. <TEXT_START> {sample} <TEXT_END>""")
    return response.strip().lower() == 'yes'

def generate_query(sample):
    response = _call_to_LLM(f"""Generate a question that could be answered
    by the following text: <TEXT_START> {sample} <TEXT_END>. The question should be placed
    between <QUERY_START> and <QUERY_END> tags.""")
    return response.split('<QUERY_START>')[1].split('<QUERY_END>')[0].strip()

def generate_synthetic_data(size):
    # 1. Load samples
    samples = load_samples(size)
    # 2. Filter samples
    filtered_samples = [sample for sample in samples if filter_sample(sample)]
    # 3. Generate queries
    data = []
    for sample in filtered_samples:
        query = generate_query(sample)
        if query and len(query) > 10:
            data.append((query, sample))

    return data



if __name__ == "__main__":
    ## Load environment variables
    env_file = '.env' # for interactive sessions change to the correct path
    config  = dotenv_values(env_file)
    for env_var in ['LOGS_FOLDER','SAVED_MODELS_DIR', 'HUGGINGFACE_TOKEN_FILE','DATA_STORAGE']:
        assert env_var in config, f'Could not find variable {env_var} in .env file: {env_file}'

    ## Setup logging
    start_time = datetime.now()
    root_logger = logging.getLogger()
    setup_logging(config, root_logger)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(1000)
    
    # Save to csv in (query, relevant_sentence) format
    writer = csv.writer(open("synthetic_data.csv", 'w', newline=''))
    writer.writerow(['query', 'relevant_sentence'])
    for query, sentence in synthetic_data:
        writer.writerow([query, sentence]) 
    writer.close()


