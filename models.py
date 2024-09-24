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
        # Truncate to first 3 layers
        self.model.config.num_hidden_layers = 3
        self.model.layers = torch.nn.ModuleList(list(self.model.layers)[:3])

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)






