#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/6 16:03
# @author: zhangcw
# @content:topic model(lsi and lda)

from gensim import corpora,models

class Topic(object):
    def __init__(self,doc_list,keyword_num,model='LSI',num_topics=4):
        self.dictionary = corpora.Dictionary(doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tiidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model(corpus)
