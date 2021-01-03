#!/usr/bin/env/ python
# -*- coding: utf-8 -*-  
# @date: 2019/9/8 15:07
# @author: zhangcw
# @content: spam classification


def get_data(ham="data/ham_data.txt",spam="data/spam_data.txt"):
    with open(ham,encoding="utf8") as f:
