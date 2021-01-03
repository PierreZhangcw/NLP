#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/26 21:16
# @author: zhangcw
# @content: preprocess the origin data as standard input

import time
import math
import re
import unicodedata
from language import Lang
from config import *

def unicodeToAscii(s):
    '''
    turn a Unicode string to plain ASCII
    :param s: string
    :return: strings in plain ASCII
    '''
    s = [c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn']
    return ''.join(s)

def normalizeString(s):
    '''
    turn string to lowercase and remove non-letter characters
    :param s: input string
    :return: string after processing
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    '''
    Loading origin data......
    :param lang1: first language
    :param lang2: second language
    :param reverse: if true, translate lang2 to lang1; else translate lang1 to lang1
    :return: object Lang(input_lang, output_lang) and translation pairs
    '''
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    '''
    see whether a sentence pair meet the following conditions
    1. sentence length if small than MAX_LENGTH 2. sentence2 startswith eng_prefixes
    :param p: sentence pair
    :return: bool value true(meet the conditions ) or false(do not meet the conditions)
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    '''
    filter pairs that don't meet the conditions
    :param pairs: all pairs to check
    :return: pairs after filtering
    '''
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    '''
    filter pairs and complete two language class
    :param lang1:
    :param lang2:
    :param reverse:
    :return:
    '''
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    '''
    use word index to replace word in sentence
    :param lang: language class
    :param sentence: sentence to index
    :return: index sequence for the sentence
    '''
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    '''
    turn a sentence to index sequence with type Tensor
    :param lang:
    :param sentence:
    :return:
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang,output_lang,pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)