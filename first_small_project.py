# -*- coding: utf-8 -*-
"""First small project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-IsKmJE8KvNuGu8WlHl02da_JTa5hHNB

## Setup
"""

# Uncomment these lines if not executing on colab
#!pip install -U transformers
#!pip install -U accelerate
#!pip install datasets

# I imported a bunch of packages that are not actually used, because this code is
# a Frankenstein monster of code snippets I found online. One day I will clean it up.

import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset, Dataset
from sklearn.model_selection import train_test_split

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling

import os, sys, copy, logging
from datetime import datetime, timedelta
from tqdm import tqdm

from datasets import load_dataset
from datasets import load_metric

# setting up logging
start_time = datetime.now()
try:
    logging.basicConfig(level=logging.DEBUG, filename=f'{os.path.dirname(os.path.abspath(__file__))}/logs/{start_time}_{os.path.basename(__file__)}.log', format= '%(asctime)s %(levelname)s : %(message)s')
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
except:
    print("assuming this is a live session, logging only to console")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

logging.info(f"{start_time} - Imports finished, starting script\n\n")

# Display info on device available
logging.debug(f"CUDA available: {torch.cuda.is_available()}")
logging.debug(f"Devices available: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")
#device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
debug_str = f"Now using device: {device}"
if device.type == 'cuda':
    debug_str += '\n\t'+torch.cuda.get_device_name(0)
    debug_str += '\n\tMemory Usage:'
    debug_str += f'\n\t\tAllocated: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB'
    debug_str += f'\n\t\tCached:    {round(torch.cuda.memory_reserved(0) /1024**3,1)} GB'

logging.debug(debug_str)
logging.debug(f'\tDefault location for tensors: {torch.rand(3).device}')
#torch.set_default_tensor_type(torch.cuda.FloatTensor) #change default tensor type -
# THIS LINE BREAKS EVERYTHING FOR SOME REASON
#logging.debug(f'\tDefault location for tensors: {torch.rand(3).device}')


# Load dataset
if os.getenv("COLAB_RELEASE_TAG"):
    # from my Google drive
    from google.colab import drive
    drive.mount('/content/drive')
    working_dir = "/content/drive/MyDrive/Unibe/"
    data_dir = working_dir
else:
    # from Ubelix container
    working_dir = '/research_storage/'
    data_dir = working_dir + 'Walser_data/'

file_name = 'Walser_data.txt'

with open(data_dir + file_name, 'r') as f:
    walser_text = f.read()
# If you want to check that the text file is accessed
#logging.info('extract text from Walser: '+walser_text[:50])

# Load model directly from huggingface's repo
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b")
metric = load_metric("accuracy")

# move it to the GPU
model.to(device)

# fix tokenizer issue
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load dataset
train_file = data_dir + "train_dataset.txt"
test_file  = data_dir + "test_dataset.txt"
dataset = load_dataset("text", data_files={"train":train_file, "test":test_file})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# move to GPU
tokenized_datasets = tokenized_datasets.map(lambda batch: {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
#tokenized_datasets.to(device) # throws error!

# Check that the model outputs something before fine-tuning
prompt = 'Once upon a time, there was a'
inputs = tokenizer(prompt, return_tensors="pt")
logging.debug(f"inputs are in CUDA: {inputs['input_ids'].is_cuda}")
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to GPU
logging.debug(f"inputs are in CUDA now: {inputs['input_ids'].is_cuda}") # inputs[0].is_cuda if bugs ?
logging.debug(f"model is in CUDA: {all(p.is_cuda for p in model.parameters())}")
outputs = model.generate(
    **inputs, max_new_tokens=100
)  # default generation config (+ 100 tokens)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#result = result.split("<end_answer>")[0].strip()
logging.info("Testing that inference works:\n" + result)

# Define a custom training loop
# def compute_loss(model, inputs):
#     outputs = model(inputs.input_ids, inputs.attention_mask, inputs=input_ids)
#     return outputs.loss

# Define training arguments
training_args = TrainingArguments(
    output_dir=working_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_total_limit=5,
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    tokenizer=tokenizer,
    #compute_metrics=metric,
)

train_start_time = datetime.now()
logging.info(f"{train_start_time} - Starting training")
#trainer.compute_loss = compute_loss
train_result = trainer.train()
train_end_time = datetime.now()
logging.info(f"{train_end_time} - Training finished !")
logging.info(f"Time spent until training starts: {train_start_time - start_time}")
logging.info(f"Time spent on training: {train_end_time - train_start_time}")

# Save the metrics of the training
# read the doc here if you can't understand the report: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.log_metrics
metrics = train_result.metrics
trainer.log_metrics("all", metrics)
trainer.save_metrics("all", metrics)

# Save the fine-tuned model
model.save_pretrained( data_dir + "fine_tuned_pythia-2.8b-Walser")
tokenizer.save_pretrained( data_dir + "fine_tuned_pythia-2.8b-Walser") #i did not change the tokenizer ?
logging.info('model saved')

# load for inference
#tokenizer = AutoTokenizer.from_pretrained(data_dir + "fine_tuned_pythia-70m-Walser")
#model = AutoModelForCausalLM.from_pretrained(data_dir + "fine_tuned_pythia-70m-Walser")
