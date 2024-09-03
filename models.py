# -*- coding: utf-8 -*-
"""
Define custom models here
"""

import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        lstm_out, _ = self.lstm(embeds)
        out = self.linear(lstm_out)
        return out








