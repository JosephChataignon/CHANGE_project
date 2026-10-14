
import sys, logging, csv
from datetime import datetime
from dotenv import dotenv_values
from any_llm import AnyLLM

from data import get_CHANGE_data_random_samples
from logging_utils import setup_logging



# Use bigger chunks than the default (chunk_size=6) so each generated question has more
# context to be answerable from. Adjust chunk_size here if needed.
SEGMENTATION_METHOD = {"method": "sentence", "chunk_size": 24, "overlap": 3}

def load_samples(sample_size: int):
    data_storage = config['DATA_STORAGE']
    assert data_storage is not None, "DATA_STORAGE environment variable not set"
    return get_CHANGE_data_random_samples(sample_size, data_storage, segmentation_method=SEGMENTATION_METHOD)

def _initialize_llm():
    provider = config.get('llm_provider')
    api_key = config.get('llm_api_key')
    api_base = config.get('llm_api_base')
    api_base = None if api_base == '' else api_base
    
    llm = AnyLLM.create(provider, api_key=api_key, api_base=api_base)
    return llm

def _call_to_LLM(llm, prompt):
    model_name = config.get('llm_model')
    try:
        response = llm.completion(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in text classification and question generation."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.warning(f"LLM call failed, skipping this sample: {e}")
        return None

def filter_sample(llm, sample):
    response = _call_to_LLM(llm, f"""Is the following text a paragraph of text 
    related to education science, or contain facts about education in general ?
    Answer with 'yes' or 'no' only, do not include any explanations or punctuation. 
    <TEXT_START> {sample} <TEXT_END>""")
    if response is None:
        return False
    return response.replace("'","").strip().lower() == 'yes'

def filter_queries(llm, query, sample):
    response = _call_to_LLM(llm, f"""Is the following question answerable from the text below ?
    Answer with 'yes' or 'no' only, do not include any explanations or punctuation. 
    <TEXT_START> {sample} <TEXT_END>
    <QUESTION_START> {query} <QUESTION_END>""")
    if response is None:
        return False
    return response.replace("'","").strip().lower() == 'yes'

def generate_query(llm, sample):
    response = _call_to_LLM(llm, f"""Generate a question that could be answered
    by the following text: <TEXT_START> {sample} <TEXT_END>. 
    The question must beself-contained and answerable from the text, but not overly specific. Do not include the answer in the question.
    Do not introduce facts that are not in the text. The wording should not be exactly the same as in the text, but it should be clear that the answer can be found in the text.
    Do not refer to the text as "the passage", "the text", or "this extract"; write a self-contained question that makes sense without the source being mentioned.
    Avoid yes/no questions.
    The question you generate should be placed between <QUESTION_START> and <QUESTION_END> tags.""")
    if response is None:
        return None
    try:
        return response.split('<QUESTION_START>')[1].split('<QUESTION_END>')[0].split('</QUESTION_END>')[0].strip() #in practice, llm sometimes adds a </QUESTION_END> tag
    except IndexError:
        logging.warning("Could not parse QUERY tags from LLM response, skipping this sample")
        return None

def generate_synthetic_data(size):
    # 1. Load samples and initialize LLM
    logging.info("starting to load text samples")
    samples = load_samples(size)
    llm = _initialize_llm()
    # 2. Filter samples
    filtered_samples = [sample for sample in samples if filter_sample(llm, sample)]
    # 3. Generate queries
    data = []
    for sample in filtered_samples:
        query = generate_query(llm, sample)
        if query and len(query) > 10:
            if filter_queries(llm, query, sample):
                data.append((query, sample))

    return data



if __name__ == "__main__":
    ## Load arguments
    try:
        size = int(sys.argv[1]) 
        filename = str(sys.argv[2])
    except (IndexError, ValueError):
        print("Usage: python script.py <size> <filename>")
        sys.exit(1)
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
    logging.info(f"Starting generation of {size} synthetic data samples...")
    synthetic_data = generate_synthetic_data(size)
    logging.info(f"Finished generation of synthetic data samples in {datetime.now() - start_time}.")
    logging.info(f"Generated {len(synthetic_data)} out of {size} synthetic data samples (difference is due to filtering incorrect samples)")
    
    # Save to csv in (query, relevant_sentence) format
    filepath = "validation_stage_2/"+filename+".csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['query', 'relevant_sentence'])
        for query, sentence in synthetic_data:
            writer.writerow([query, sentence]) 
    logging.info("Saved synthetic data to csv file: " + filepath)
