import os, sys, logging
from dotenv import load_dotenv, dotenv_values
from datetime import datetime, timedelta

from logging_utils import setup_logging, display_CUDA_info
from data import get_CHANGE_data

## Load environment variables
env_file = '.env' # for interactive sessions change to the correct path
config  = dotenv_values(env_file)
assert 'LOGS_FOLDER' in config, f'Could not find variable LOGS_FOLDER in .env file: {env_file}'
assert 'SAVED_MODELS_DIR' in config, f'Could not find variable SAVED_MODELS_DIR in .env file: {env_file}'

start_time = datetime.now()

date_str = start_time.isoformat()[:19]
log_file = f"{config['LOGS_FOLDER']}/{date_str}_{os.path.basename(__file__)}_TEEEEEEEEST.log"
root_logger = logging.getLogger()
setup_logging(log_file, root_logger)


logging.info(f"{start_time} - Imports finished, starting script\n\n")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)


train_file, test_file = get_CHANGE_data('Walser')


logging.info(f'test & train files:{train_file},{test_file}')
