# -*- coding: utf-8 -*-
"""
Launcher file
"""

## Imports
import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset, Dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os, sys, copy, logging
#from transformers.utils import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values

from datasets import load_dataset
from datasets import load_metric

# in interactive sessions, uncomment this line:
#sys.path.insert(0, r'/path/to/code/folder')
from logging_utils import setup_logging, display_CUDA_info, print_trainable_parameters
from data import get_CHANGE_data

## Load environment variables
env_file = '.env' # for interactive sessions change to the correct path
config  = dotenv_values(env_file)
assert 'LOGS_FOLDER' in config, f'Could not find variable LOGS_FOLDER in .env file: {env_file}'
assert 'SAVED_MODELS_DIR' in config, f'Could not find variable SAVED_MODELS_DIR in .env file: {env_file}'

## Setup logging
start_time = datetime.now()
date_str = start_time.isoformat()[:19]
log_file = f"{config['LOGS_FOLDER']}/{date_str}_{os.path.basename(__file__)}.log"
root_logger = logging.getLogger()
transformers_logger = transformers.logging.get_logger()
setup_logging(log_file, root_logger, transformers_logger)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)

logging.info("Setup finished, starting script\n\n")



## Set parameters

# get data files ("Walser" or "Max-Planck" or "Max-Planck-test")
data_set = 'walser'

## Chose model
# model_name = "openai-gpt"
model_name = "EleutherAI/pythia-410m"

# load and fix tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))


## Training configuration

# Bitsandbytes quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=bnb_config,
    device_map="auto",
    use_cache = False,
    trust_remote_code=True,
)
# LoRA
qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    task_type="CAUSAL_LM"
)
# GPTQ
quantization_config = GPTQConfig(
    bits=4,
    dataset = "ptb", # default is "c4" for calibration dataset
    tokenizer=tokenizer
)

## For quantization with GPTQ (no training afterward, inference only)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

## Simple loading
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)




# set name where the trained model will be saved
instance_name = f"{model_name.replace('/','-')}_finetuned-on_{data_set}_{start_time.date()}"
logging.info(f'Model loaded: {model_name}')
display_CUDA_info(device)
logging.info(f'Output (fine-tuned) model will be saved with the name: {instance_name}')


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
def inference_test(prompt,model,tokenizer,device):
    inputs = tokenizer(prompt, return_tensors="pt")
    logging.debug(f"Inference: inputs are in CUDA: {inputs['input_ids'].is_cuda}")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to GPU
    logging.debug(f"Inference: inputs are in CUDA now: {inputs['input_ids'].is_cuda}") # inputs[0].is_cuda if bugs ?
    logging.debug(f"Inference: model is in CUDA: {all(p.is_cuda for p in model.parameters())}")
    outputs = model.generate(
        **inputs, max_new_tokens=100
    )  # default generation config (+ 100 tokens)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return result

result = inference_test('Once upon a time, there was a',model,tokenizer,device)
logging.info("Testing that inference works:\n" + result)


## for Accelerate use
accelerator = Accelerator()

# Define training arguments
training_args = TrainingArguments(
    output_dir=config['SAVED_MODELS_DIR'],
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Set this to match DeepSpeed's train_micro_batch_size_per_gpu
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    gradient_accumulation_steps=2,  # Set this to match DeepSpeed's gradient_accumulation_steps
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=True,  # Enable BF16 to match DeepSpeed's bf16 setting
    logging_steps=20,
    max_steps=100,
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

# load for inference
#tokenizer = AutoTokenizer.from_pretrained(config['SAVED_MODELS_DIR'] + "fine_tuned_pythia-70m-Walser")
#model = AutoModelForCausalLM.from_pretrained(config['SAVED_MODELS_DIR'] + "fine_tuned_pythia-70m-Walser")
