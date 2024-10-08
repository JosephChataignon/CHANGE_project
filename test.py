import os, sys, logging, torch, transformers
from dotenv import load_dotenv, dotenv_values
from datetime import datetime, timedelta
#from datasets import load_metric

from logging_utils import setup_logging, display_CUDA_info
from data import get_CHANGE_data

## Load environment variables
env_file = '.env' # for interactive sessions change to the correct path
config  = dotenv_values(env_file)
assert 'LOGS_FOLDER' in config, f'Could not find variable LOGS_FOLDER in .env file: {env_file}'
assert 'SAVED_MODELS_DIR' in config, f'Could not find variable SAVED_MODELS_DIR in .env file: {env_file}'

start_time = datetime.now()

date_str = start_time.isoformat()[:19]
root_logger = logging.getLogger()
transformers_logger = transformers.logging.get_logger()
setup_logging(config, root_logger, transformers_logger)


logging.info(f"{start_time} - Imports finished, starting script\n\n")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)


train_file, test_file = get_CHANGE_data('Walser')


logging.info(f'test & train files:{train_file},{test_file}')

#metric = load_metric("accuracy")
