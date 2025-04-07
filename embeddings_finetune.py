# -*- coding: utf-8 -*-
"""
Launcher file for the fine-tuning of embeddings models
"""

################################ Imports ################################
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, Dataset as TorchDataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling
from sentence_transformers import SentenceTransformer, losses, InputExample, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os, sys, copy, logging, random
import nltk
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values

from datasets import Dataset as HFDataset, load_dataset

# in interactive sessions, uncomment this line:
#sys.path.insert(0, r'/path/to/code/folder')
from logging_utils import setup_logging, display_CUDA_info, print_trainable_parameters, get_tb_callback, inference_test
from data import get_CHANGE_data, get_CHANGE_data_for_sentences
from models import load_model

################################### SETUP ######################################
## Load environment variables
env_file = '.env' # for interactive sessions change to the correct path
config  = dotenv_values(env_file)
for env_var in ['LOGS_FOLDER','SAVED_MODELS_DIR', 'HUGGINGFACE_TOKEN_FILE', 'DATA_STORAGE']:
    assert env_var in config, f'Could not find variable {env_var} in .env file: {env_file}'
# extract Huggingface token
with open(config['HUGGINGFACE_TOKEN_FILE'], 'r') as file:
    hf_token = file.read().strip()
    config['HF_TOKEN'] = hf_token

## Setup logging
start_time = datetime.now()
root_logger = logging.getLogger()
transformers_logger = transformers.logging.get_logger()
setup_logging(config, root_logger, transformers_logger)

## Setup CUDA
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)

logging.info("Setup finished, starting script\n\n")

############################### LOADING MODEL, TOKENIZER AND DATA ###############################

# get data files ("education" or "education_sample" ...)
data_set = 'education_sample'

# Chose model (examples: "Lajavaness/bilingual-embedding-large", "sentence-transformers/all-mpnet-base-v2"...)
model_name = "sentence-transformers/all-mpnet-base-v2"

model = SentenceTransformer(model_name, device=f'cuda:{local_rank}')
model = DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
model.parallel_training = False


# set name where the trained model will be saved
instance_name = f"{model_name.replace('/','-')}_finetuned-on_{data_set}_{start_time}"
logging.info(f'Model loaded: {model_name}')
logging.info(model)
logging.info(f'Output (fine-tuned) model will be saved with the name: {instance_name}')
display_CUDA_info(device)


## LOAD DATASET
logging.info(f'Loading data set: {data_set}')
dataset = get_CHANGE_data_for_sentences(data_set, config['DATA_STORAGE'])
#dataset = load_dataset("sentence-transformers/all-nli", "triplet")
train_dataset = dataset["train"])
eval_dataset = dataset["dev"]
test_dataset = dataset["test"]



################################# TRAINING ########################################
logging.info("Starting training")

# for TensorBoard logging
tensorboard_callback = get_tb_callback(config,instance_name)

loss = losses.MultipleNegativesRankingLoss(model).to(torch.device(f'cuda:{local_rank}'))

# Optional: Create an evaluator
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name=data_set,
)
dev_evaluator(model.module)

# Run the training
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=os.path.join(config['SAVED_MODELS_DIR'],'embedding-finetune-test'),
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    gradient_accumulation_steps=2,  # Effectively same batch size but less memory
    dataloader_num_workers=0,       # Disable parallel data loading
    dataloader_pin_memory=False,    # Disable pinned memory
)
trainer = SentenceTransformerTrainer(
    model=model.module,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

logging.info(f"CUDA available: {torch.cuda.is_available()}")
logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
logging.info(f"Device count: {torch.cuda.device_count()}")

display_CUDA_info(device)
train_start_time = datetime.now()
logging.info(f"{train_start_time} - Starting training")
try:
    trainer.train()
    train_end_time = datetime.now()
    logging.info(f"{train_end_time} - Training finished !")
    display_CUDA_info(device)
except Exception as e:
    train_end_time = datetime.now()
    logging.error(f"{train_end_time} - Training failed")
    logging.info(f"Time spent until training starts: {train_start_time - start_time}")
    logging.info(f"Time spent on training: {train_end_time - train_start_time}")
    display_CUDA_info(device)
    raise

# log some of the time spent
logging.info(f"Time spent until training starts: {train_start_time - start_time}")
logging.info(f"Time spent on training: {train_end_time - train_start_time}")

# Save the fine-tuned model
model.module.save_pretrained(f"{config['SAVED_MODELS_DIR']}/{instance_name}")
logging.info(f"model saved at {config['SAVED_MODELS_DIR']}/{instance_name}")
