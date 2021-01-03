#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/6 16:03
# @author: zhangcw
# @content:topic model(lsi and lda)

import math
import functools
from gensim import corpora,models
import sys
sys.path.append("..")
from utility.utility import *

class TopicModel(object):
    def __init__(self,doc_list,keyword_num,model='LSI',num_topics=4):

        # construct word space
        self.dictionary = corpora.Dictionary(doc_list)

        # trans doc_list to vector representation using BOW model
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]

        # get tfidf vector for the corpus
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics

        # select the model
        if model.lower() == "lsi":
            self.model = models.LsiModel(self.corpus_tfidf,id2word=self.dictionary,num_topics=self.num_topics)
        else:# model llda
            self.model = models.LdaModel(self.corpus_tfidf,id2word=self.dictionary,num_topics=self.num_topics)

        # get words set of the corpus
        word_dic = get_words_set(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def get_wordtopic(self,word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self,word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # cosin similarity
        def calsim(l1,l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

            # similarity of each word with input doc
        sim_dic = {}
        for k,v in self.wordtopic_dic.items():
            if k in word_list:
                sim = calsim(v,senttopic)
                sim_dic[k] = sim

        for k,v in sorted(sim_dic.items(),key=functools.cmp_to_key(cmp),reverse=True)[:self.keyword_num]:
            print(k+"/",end = '')
        print()



