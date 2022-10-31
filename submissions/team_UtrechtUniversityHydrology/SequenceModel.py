from turtle import forward
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(self,
                 in_size: int,
                 hidden_size: int,
                 n_lstm: int,
                 out_size: int,
                 dropout_rate: float,
                 cuda: bool = False):
        super(SequenceModel, self).__init__()

        self.linear_in = nn.Linear(in_features=in_size,
                            out_features=hidden_size)
        self.dropout_in = nn.Dropout(p = dropout_rate)
        self.lstm = nn.LSTM(input_size = hidden_size,
                    hidden_size = hidden_size,
                    num_layers = n_lstm,
                    dropout  = dropout_rate)
        self.dropout_out = nn.Dropout(p = dropout_rate)
        self.linear_out = nn.Linear(in_features=hidden_size,
                            out_features=out_size)
        
        if cuda and torch.cuda.is_available():
            self.cuda()
            
    def forward(self,
                x: torch.Tensor):
        x = self.linear_in(x)
        x = self.dropout_in(x)
        y, _ = self.lstm(x)
        y = self.dropout_out(y)
        y = self.linear_out(y)
        return y