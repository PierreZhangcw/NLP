#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/1 13:45
# @author: zhangcw
# @content: tfidf model

import math

class TfIdf:
    def __init__(self,doc_list,keyword_num=10):

        self.doc_list = doc_list
        self.keyword_num = keyword_num # number of keywords return for each document

        self.words_set = self.get_words_set()
        self.idf_dic,self.default_idf = self.get_idf_dic()

    def get_words_set(self):
        '''
        static all words in doc_list
        :return: set of words
        '''
        words_set = set()
        for doc in self.doc_list:
            words_set.update(doc)
        return words_set

    def get_idf_dic(self):
        '''
        calculate idf value for each word
        :return: dictionary of idf value for each word
        '''
        idf_dic = dict([word,0] for word in self.words_set)
        for doc in self.doc_list:
            for word in set(doc):
                idf_dic[word]+=1

        D = len(self.doc_list)
        default_idf = math.log(D) # default idf value for word that not appear in corpus
        # calculate idf value for each word with plus 1 smooth
        for word in idf_dic:
            idf_dic[word] = math.log(D/(idf_dic[word]+1))
        return idf_dic,default_idf

    def get_tf_dic(self,doc):
        '''
        calculate the tf value for each word in a document
        :param doc: target document
        :return: tf value dictionary for target document
        '''
        tf_dic = {}
        for word in doc:
            tf_dic[word] = tf_dic.get(word,0) + 1
        # calculate tf value for each word in this document
        words_count = len(doc)
        for word in tf_dic:
            tf_dic[word] = tf_dic[word]/words_count
        return tf_dic

    def get_tf_idf(self,doc):
        '''
        calculate the tfidf value for each word in a document
        :param doc: target document(words list)
        :return: tfidf dictionary {word:tfidf value,...}
        '''
        tf_dic = self.get_tf_dic(doc)
        tfidf_dic = {}
        for word in tf_dic:
            tf = tf_dic[word]
            idf = self.idf_dic.get(word,self.default_idf)
            tfidf_dic[word] = tf*idf
        return tfidf_dic

    def get_key_words(self,doc,keyword_num=None):
        '''
        get keywordnum key words of doc
        :param doc: target document
        :param keyword_num: number of key words we want
        :return: key words list
        '''
        if not keyword_num:
            keyword_num = self.keyword_num
        tfidf_dic = self.get_tf_idf(doc)
        key_words = []
        for item in sorted(tfidf_dic.items(),key=lambda x:x[1],reverse=True)[:keyword_num]:
            key_words.append(item[0])
        return key_words

    def get_all_key_words(self):
        '''
        get key words for each doc in doc list
        :return: list of key words list
        '''
        all_key_words = []
        for doc in self.doc_list:
            all_key_words.append(self.get_key_words(doc))
        return all_key_words