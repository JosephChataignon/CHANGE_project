# -*- coding: utf-8 -*-
"""
Launcher file
"""

################################ Imports ################################
import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset, Dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling
from accelerate import Accelerator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os, sys, copy, logging
#from transformers.utils import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values

from datasets import load_dataset

# in interactive sessions, uncomment this line:
#sys.path.insert(0, r'/path/to/code/folder')
from logging_utils import setup_logging, display_CUDA_info, print_trainable_parameters, get_tb_callback, inference_test
from data import get_CHANGE_data
from models import load_model

################################### SETUP ######################################
## Load environment variables
env_file = '.env' # for interactive sessions change to the correct path
config  = dotenv_values(env_file)
for env_var in ['LOGS_FOLDER','SAVED_MODELS_DIR', 'HUGGINGFACE_TOKEN_FILE']:
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)

logging.info("Setup finished, starting script\n\n")



############################### LOADING MODEL, TOKENIZER AND DATA ###############################

# get data files ("Walser" or "Max-Planck" or "Max-Planck-test")
data_set = 'Walser'

# Chose model (examples: "openai-gpt", "EleutherAI/pythia-410m", "truncatedLlama2")
model_name = "truncatedLlama2"

model, tokenizer = load_model(model_name, config)
model.to(device)


# set name where the trained model will be saved
instance_name = f"{model_name.replace('/','-')}_finetuned-on_{data_set}_{start_time}"
logging.info(f'Model loaded: {model_name}')
logging.info(model)
logging.info(f'Output (fine-tuned) model will be saved with the name: {instance_name}')
display_CUDA_info(device)


## Load and tokenize dataset
logging.info(f'Loading data set: {data_set}')
dataset = get_CHANGE_data(data_set)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# move to GPU
tokenized_datasets = tokenized_datasets.map(lambda batch: {k: v.to(device).long() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
display_CUDA_info(device)


# Check that the model outputs something before fine-tuning
result = inference_test('Once upon a time, there was a',model,tokenizer,device)
logging.info("Testing that inference works:\n" + result)





################################# TRAINING ########################################

# for Accelerate use
accelerator = Accelerator()
# for TensorBoard logging
tensorboard_callback = get_tb_callback(config,instance_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir=config['SAVED_MODELS_DIR'],
    logging_dir=config['LOGS_FOLDER'],
    overwrite_output_dir=True,
    #remove_unused_columns=False, # fix issue when reusing Llama layers BUT BREAKS EVERYTHING
    per_device_train_batch_size=4,  # Set this to match DeepSpeed's train_micro_batch_size_per_gpu
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=2,  # Set this to match DeepSpeed's gradient_accumulation_steps
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=True,  # Enable BF16 to match DeepSpeed's bf16 setting
    logging_steps=100,
    logging_strategy="steps",
    run_name=instance_name,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    tokenizer=tokenizer,
    callbacks=[tensorboard_callback],
)

## for QLoRA training
# trainer = SFTTrainer(
#     model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['test'],
#     tokenizer=tokenizer,
#     peft_config=qlora_config,
#     dataset_text_field="text",
#     max_seq_length=2048,
# )

## for Accelerate use
trainer = accelerator.prepare(trainer)



display_CUDA_info(device)
train_start_time = datetime.now()
logging.info(f"{train_start_time} - Starting training")
try:
    #trainer.compute_loss = compute_loss
    train_result = trainer.train()
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

# Save the metrics of the training
# read the doc here if you can't understand the report: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.log_metrics
metrics = train_result.metrics
trainer.log_metrics("all", metrics)
trainer.save_metrics("all", metrics)

# Save the fine-tuned model
model.save_pretrained(f"{config['SAVED_MODELS_DIR']}/{instance_name}")
tokenizer.save_pretrained(f"{config['SAVED_MODELS_DIR']}/{instance_name}") #i did not change the tokenizer ?
logging.info(f"model saved at {config['SAVED_MODELS_DIR']}/{instance_name}")



# In order to re-load the models for inference
#tokenizer = AutoTokenizer.from_pretrained(config['SAVED_MODELS_DIR'] + "fine_tuned_pythia-70m-Walser")
#model = AutoModelForCausalLM.from_pretrained(config['SAVED_MODELS_DIR'] + "fine_tuned_pythia-70m-Walser")
