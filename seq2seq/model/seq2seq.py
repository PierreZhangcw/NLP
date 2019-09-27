#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/24 10:35
# @author: zhangcw
# @content:
from __future__ import unicode_literals, print_function, division
from io import open

import string
import time
import random

import torch
from torch import optim
from encoder import *
from decoder import *
from config import *
from utility import *


def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,max_length = MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length,encoder.hidden_size,device = device)

    loss = 0

    for ei in range(input_length):
        # shape[1,1,hidden_size], shape[1,1,hidden_size]
        encoder_output,encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[0,0] # shape[1,hidden_size]

    decoder_input = torch.tensor([[SOS_token]],device=device)
    decoder_hidden = encoder_hidden

    user_teacher_forcing = True if random.random()<teacher_forcing_ratio else False
    if user_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output,target_tensor[di])
            decoder_input = target_tensor[di]


    else:
        for di in range(target_length):
            decoder_output,decoder_hidden,decoder_attention = decoder(
                decoder_input,decoder_hidden,encoder_outputs)
            topv,topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output,target_tensor[di])
            if decoder_input.item()==EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def trainIter(input_lang,output_lang,pairs, encoder, decoder, n_iters, print_every=1000,plot_every=100,learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
    training_pairs = [tensorsFromPair(input_lang,output_lang,random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1,n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor, target_tensor = training_pair

        loss = train(input_tensor,target_tensor,
                     encoder,decoder,
                     encoder_optimizer,decoder_optimizer,
                     criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter%print_every==0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print("%s (%d %d%%) %.4f"%(timeSince(start,iter/n_iters),iter,
                                       iter/n_iters*100,print_loss_avg))

        if iter%plot_every==0:
            plot_loss_avg = plot_loss_total/plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        #showPlot(plot_losses)



if __name__=='__main__':
    input_lang,output_lang,pairs = prepareData('eng','fra',reverse=True)
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words,hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size,output_lang.n_words,dropout_p=0.1).to(device)

    trainIter(input_lang,output_lang,pairs,encoder,decoder,75000,print_every=5000)
