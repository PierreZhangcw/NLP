#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/26 22:06
# @author: zhangcw
# @content: simple decoder and decoder with attention

import torch.nn as nn
import torch.nn.functional as F
from config import *

class DecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input:torch.TensorLong,hidden:shape[1,1,hidden_size]

        output = self.embedding(input).view(1,1,-1) # shape[1,1,hidden_size]
        output = F.relu(output)  # shape[1,1,hidden_size]
        output,hidden = self.gru(output,hidden)   # shape[1,1,hidden_size],shape[1,1,hidden_size]
        output = self.softmax(self.out(output[0]))  # shape[1,output_size]
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device = device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size   # dimension of vector
        self.output_size = output_size   # total number of words for output language
        self.dropout_p = dropout_p
        self.max_length = max_length     # maximum length of one sentence

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input:torch.TensorLong,hidden:shape[1,1,hidden_size],
        # encoder_outputs:shape[max_length,hidden_size]

        embedded = self.embedding(input).view(1, 1, -1) # shape[1,1,hidden_size]
        embedded = self.dropout(embedded) # shape[1,1,hidden_size]

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)  # shape[1,max_length]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), # bmm(shape[1,1,max_length],shape[1,max_length,hidden_size]) = shape[1,1,hidden_size]
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1) # shape[1,2*hidden_size]
        output = self.attn_combine(output).unsqueeze(0)  # shape[1,1,hidden_size]

        output = F.relu(output)   # shape[1,1,hidden_size]
        output, hidden = self.gru(output, hidden) # gru(shape[1,1,hidden_size],shape[1,1,hidden_size]) = (shape[1,1,hidden_size],shape[1,1,hidden_size])

        output = F.log_softmax(self.out(output[0]), dim=1)
        # log_softmax(out(shape[1,hidden_size]),dim=1) = log_softmax(shape[1,output_size],dim=1) = shape[1,output_size]
        return output, hidden, attn_weights # shape[1,output_size], shape[1,1,hidden_size], shape[1,max_length]

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)