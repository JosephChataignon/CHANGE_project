# -*- coding: utf-8 -*-
"""
Define custom models here
"""

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import BitsAndBytesConfig, GPTQConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model(model_name, config, tokenizer_name=None):
    '''Load a model based on model_name. The model ca be from the Huggingface Hub
    or defined in this file'''

    # chose tokenizer
    if model_name.lower() == 'truncatedllama2':
        tokenizer_name = "meta-llama/Llama-2-7b-hf"
    else:
        tokenizer_name = model_name

    assert model_name != None, "Error: no specified model name"
    assert tokenizer_name != None, "Error: no specified tokenizer"
    
    # load and fix tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,use_auth_token=config['HF_TOKEN'])



    ## Training configuration

    # Bitsandbytes quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Load base model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     #quantization_config=bnb_config,
    #     device_map="auto",
    #     use_cache = False,
    #     trust_remote_code=True,
    # )
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
        dataset = "c4", # default is "c4" for calibration dataset
        tokenizer=tokenizer
    )

    ## For quantization with GPTQ (no training afterward, inference only)
    # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

    ## Simple loading
    if model_name.lower() == 'truncatedllama2':
        model = truncatedLlama2(id_token=config['HF_TOKEN'])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    #fix tokenizer empty pad token
    if tokenizer.pad_token is None:
        #tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer




class truncatedLlama2(torch.nn.Module):
    '''Takes the first 3 layers of Llama2'''
    def __init__(self, id_token):
        super(truncatedLlama2, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=id_token, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=id_token)
        # Truncate to first 3 layers
        self.model.config.num_hidden_layers = 3
        self.model.model.layers = torch.nn.ModuleList(list(self.model.model.layers)[:3])

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)






