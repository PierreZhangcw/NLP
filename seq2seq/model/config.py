#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/26 21:34
# @author: zhangcw
# @content: some configuration
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

teacher_forcing_ratio = 0.5