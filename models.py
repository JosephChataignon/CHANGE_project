# -*- coding: utf-8 -*-
"""
Define custom models here
"""

import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

class truncatedLlama2(torch.nn.Module):
    '''Takes the first 3 layers of Llama2'''
    def __init__(self, id_token):
        super(truncatedLlama2, self).__init__()
        self.model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=id_token, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=id_token)
        #self.model = torch.nn.Sequential(*list(self.model.children())[:3])  # Slice the model after the 3rd layer
        self.model.layers = torch.nn.ModuleList(list(self.model.layers)[:3])
        #self.fc = torch.nn.Linear(self.model.config.hidden_size, 4096)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        hidden_states = outputs.hidden_states[3]  # Get the output of the 3rd layer
        logits = self.fc(hidden_states)
        return logits








