import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset, Dataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, GPTQConfig
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os, sys, copy, logging
#from transformers.utils import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values

from datasets import load_dataset
from datasets import load_metric

from peft import LoraConfig
from logging_utils import setup_logging, display_CUDA_info, print_trainable_parameters
from data import get_CHANGE_data

model_name = "EleutherAI/pythia-70m"

dataset = get_CHANGE_data("walser")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


qlora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    task_type="CAUSAL_LM"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)

training_args = TrainingArguments(
    output_dir="~", 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=20,
    logging_strategy="steps",
    max_steps=100,
    run_name="test-minimal-lora"
)

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=qlora_config,
    dataset_text_field="text",
    max_seq_length=2048,
)

supervised_finetuning_trainer.train()






