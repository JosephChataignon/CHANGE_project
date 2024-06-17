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

# Bitsandbytes quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache = False
)
# LoRA
#model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model) #peft function
loraconfig = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, loraconfig)
print_trainable_parameters(model)


batch_size = 4
args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
    output_dir="~/CHANGE_project",
)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    #data_collator=collate_fn,
)
trainer.train()
