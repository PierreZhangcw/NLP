#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/24 12:35
# @author: zhangcw
# @content: encoder

import torch.nn as nn
from config import *

class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)

    def forward(self, input, hidden):
        # input:TensorLong, hidden:shape[1,hidden_size]

        embedded = self.embedding(input).view(1,1,-1) # shape[1,1,hidden_size]
        output = embedded
        output, hidden = self.gru(output,hidden)  # shape[1,1,hidden_size],shape[1,1,hidden_size]
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device = device)

