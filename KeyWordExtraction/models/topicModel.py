#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/6 16:03
# @author: zhangcw
# @content:topic model(lsi and lda)

import math
import functools
from gensim import corpora,models
from ..utility.utility import *

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
            l12 = l1[0]*l2[0]+l1[1]*l2[1]
            t1 = l1[0]**2+l1[1]**2
            t2 = l2[0]**2+l2[1]**2
            if t1*t2==0:
                return 0
            else:
                return  l12/math.sqrt(t1*t2)

            # similarity of each word with input doc
        sim_dic = {}
        for k,v in self.wordtopic_dic.items():
            if k in word_list:
                sim = calsim(v,senttopic)
                sim_dic[k] = sim

        for k,v in sorted(tfidf_dic.items(),key=functools.cmp_to_key(cmp),reverse=True)[:self.keyword_num]:
            print(k+"/",end = '')
        print()



