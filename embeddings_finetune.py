# -*- coding: utf-8 -*-
"""
Launcher file for the fine-tuning of embeddings models
"""

################################ Imports ################################
import torch
import torch.nn.functional as F
import torch.backends.cuda as cuda
from torch.utils.data import DataLoader, IterableDataset, Dataset as TorchDataset
from sklearn.model_selection import train_test_split

import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, LineByLineTextDataset, TextDataset, DataCollatorForLanguageModeling
from sentence_transformers import SentenceTransformer, losses, InputExample
from accelerate import Accelerator
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import os, sys, copy, logging, random
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
#from transformers.utils import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values
from peft import LoraConfig

from datasets import Dataset as HFDataset, load_dataset

# in interactive sessions, uncomment this line:
#sys.path.insert(0, r'/path/to/code/folder')
from logging_utils import setup_logging, display_CUDA_info, print_trainable_parameters, get_tb_callback, inference_test
from data import get_CHANGE_data
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logs CUDA info with DEBUG level
display_CUDA_info(device)

logging.info("Setup finished, starting script\n\n")



############################### LOADING MODEL, TOKENIZER AND DATA ###############################

# get data files ("education" or "education_sample" ...)
data_set = 'education_sample'

# Chose model (examples: "Lajavaness/bilingual-embedding-large", "sentence-transformers/all-mpnet-base-v2"...)
model_name = "sentence-transformers/all-mpnet-base-v2"

model = SentenceTransformer(model_name)
model.to(device)


# set name where the trained model will be saved
instance_name = f"{model_name.replace('/','-')}_finetuned-on_{data_set}_{start_time}"
logging.info(f'Model loaded: {model_name}')
logging.info(model)
logging.info(f'Output (fine-tuned) model will be saved with the name: {instance_name}')
display_CUDA_info(device)


## Load dataset
logging.info(f'Loading data set: {data_set}')
dataset = get_CHANGE_data(data_set, config['DATA_STORAGE'])

def segment_documents(examples):
    all_sentences = []
    doc_ids = []
    tokenizer = PunktSentenceTokenizer()
    for i, text in enumerate(examples["text"]):
        sentences = tokenizer.tokenize(text)
        all_sentences.extend(sentences)
        doc_ids.extend([i] * len(sentences))
    
    return {"sentence": all_sentences, "doc_id": doc_ids}

# Process the dataset to extract sentences
sentence_dataset = dataset.map(
    segment_documents, 
    batched=True, 
    remove_columns=dataset.column_names
)
logging.info(f"Dataset documents are chunked, now we have {len(sentence_dataset)} sentences.")

# Create training pairs for contrastive learning
def create_pairs(dataset, batch_size=1000):
    sentences = dataset["sentence"]
    doc_ids = dataset["doc_id"]
    pairs = []
    
    # Strategy: Sentences from same document are positive pairs
    # Sentences from different documents are negative pairs
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_doc_ids = doc_ids[i:i+batch_size]
        
        for j in range(len(batch_sentences)):
            # Find positive examples (from same document)
            pos_indices = [k for k in range(len(batch_sentences)) 
                          if batch_doc_ids[k] == batch_doc_ids[j] and k != j]
            
            if pos_indices:
                pos_idx = random.choice(pos_indices)
                
                # Find negative examples (from different documents)
                neg_indices = [k for k in range(len(batch_sentences)) 
                              if batch_doc_ids[k] != batch_doc_ids[j]]
                
                if neg_indices:
                    neg_idx = random.choice(neg_indices)
                    
                    pairs.append({
                        "anchor": batch_sentences[j],
                        "positive": batch_sentences[pos_idx],
                        "negative": batch_sentences[neg_idx]
                    })
    
    return HFDataset.from_list(pairs)

# Create pairs dataset
pairs_dataset = create_pairs(sentence_dataset)

# Reformat for MultipleNegativeRankingLoss
def format_for_mnrl(examples):
    return {
        "query": examples["anchor"],
        "positive": examples["positive"],
        # For MNRL, the negatives in the batch will be used
        # But you can also explicitly provide them
        "negative": examples["negative"]
    }

mnrl_dataset = pairs_dataset.map(format_for_mnrl)

# Split and prepare for training
train_test_split = mnrl_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


def collate_fn(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]
    return {"query": query, "positive": positive, "negative": negative}

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn
)

def convert_to_sentence_transformer_format(dataset):
    examples = []
    logging.info(f'dataset:\n{dataset}')
    for item in dataset:
        logging.info(f'\titem: {item}')
        # For MultipleNegativesRankingLoss
        examples.append(InputExample(texts=[item.get('query'), item.get('positive')]))
        # For ContrastiveLoss, uncomment below instead
        # examples.append(InputExample(texts=[item.get('query'), item.get('positive')], label=1.0))
        # examples.append(InputExample(texts=[item.get('query'), item.get('negative')], label=0.0))
    return examples

# move to GPU
# tokenized_datasets = tokenized_datasets.map(lambda batch: {k: v.to(device).long() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
# display_CUDA_info(device)





################################# TRAINING ########################################
logging.info("Starting training")

# for Accelerate use
accelerator = Accelerator()
# for TensorBoard logging
tensorboard_callback = get_tb_callback(config,instance_name)

train_loss = losses.MultipleNegativesRankingLoss(model)

train_examples = convert_to_sentence_transformer_format(train_dataset)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Step 4: Train the model
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Optional: Create an evaluator
dev_examples = convert_to_sentence_transformer_format(test_dataset[:1000])  # Limit size for evaluation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples)

# Configure training parameters
warmup_steps = int(len(train_dataloader) * 0.1)  # 10% of training data for warmup
output_path = "./finetuned-sentence-embedding-model"

# Run the training




display_CUDA_info(device)
train_start_time = datetime.now()
logging.info(f"{train_start_time} - Starting training")
try:
    #trainer.compute_loss = compute_loss
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=3,  # Adjust based on your dataset size and needs
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )
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
model.save_pretrained(f"{config['SAVED_MODELS_DIR']}/{instance_name}")
logging.info(f"model saved at {config['SAVED_MODELS_DIR']}/{instance_name}")


