#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/1 13:44
# @author: zhangcw
# @content: some util functions for processing text data

import jieba
import jieba.posseg as psg
import numpy as np

def get_stop_word(path='data/stopword.txt'):
    '''
    get stop words list
    :param path: path of stop words file
    :return: the list of stop words
    '''
    stop_words = [w.strip('\n') for w in open(path,encoding='utf8').readlines()]
    return stop_words

def seg_to_list(sentence,pos=False):
    '''
    cut sentence into words
    :param sentence: target sentence
    :param pos: whether use part-of-speech tagging
    :return: the list of (words) or (words and their tag) for sentence
    '''
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list

def word_filter(seg_list,pos=False):
    '''
    remove useless words
    :param seg_list: the list of (words) or (words and their tag) for sentence
    :param pos: whether the seg_list contains their tag
    :return: the list of words after filtering
    '''
    stop_words = get_stop_word()
    if pos:
        filter_list = []
        for word,tag in seg_list:
            if tag.startwith('n') and word not in stop_words:
                filter_list.append(word)
    else:
        filter_list = [word for word in seg_list if word not in stop_words]
    return filter_list

def load_data(pos=False,corpus_path='data/corpus.txt'):
    '''
    load corpus,process each document into list of words
    :param corpus_path: path of corpus file
    :param pos:whether the seg_list contains their tag
    :return: list of words list representation of each document
    '''
    doc_list = []
    for line in open(corpus_path,encoding='utf8').readlines():
        sentence = line.strip('\n')
        seg_list = seg_to_list(sentence,pos)
        filter_list = word_filter(seg_list,pos)
        doc_list.append(filter_list)
    return doc_list

def get_words_set(doc_list):
    '''
    static all words in doc_list
    :return: set of words
    '''
    words_set = set()
    for doc in doc_list:
        words_set.update(doc)
    return words_set

def cmp(e1,e2):
    res = np.sign(e1[1]-e2[1])
    if res!=0:
        return res
    else:
        a = e1[0]+e2[0]
        b = e2[0]+e1[0]
        if a>b:
            return 1
        elif a==b:
            return 0
        else:
            return -1