#!/usr/bin/env/ python
# -*- coding: utf8 -*-
'''
@date: 2021/01/01 10:47:03
@author: zhangcunwang
@content: 
'''
import json
from tqdm import tqdm
from utils import *


def parse_name(category):
    dic = {}
    url = "http://www.manmankan.com/dy2013/mingxing/{}/index_{}.shtml"
    page_num = 1
    print("for category '{}':".format(category))
    print("page number:", end = " ")
    while page_num < 50:
        selector = get_selector(url.format(category,page_num))
        if selector['status'] == 0:
            print("status 0 !")
            break
        else:
            selector = selector['selector']
            names = selector.xpath('//*[@class="i_cont"]/a/text()')
            urls = selector.xpath('//*[@class="i_cont"]/a//@href')
            if len(names) == 0 and page_num>2:
                print("len(names)==0!",type(names),names)
                break
            for i in range(len(names)):
                dic[urls[i]] = names[i]
            page_num += 1
        print(page_num-1, end=" ")
    return dic

def get_all_category(save_path = "../data/url_name.json"):
    dic = {}
    for i in range(65,91):
        category = chr(i)
        dic.update(parse_name(category))
    print("total names:",len(dic))
    with open(save_path,'w') as f:
        json.dump(dic,f)
    

def get_detail(url_path="../data/url_name.json",save_path="../data/star_info.json"):
    dic = {}
    url_root = "http://www.manmankan.com{}"
    with open(url_path,'r',encoding='utf8') as f:
        url_name = json.load(f)
    for k in tqdm(url_name,ncols=80):
        dic_local = {}
        dic['url'] = url_root.format(k)
        dic['name'] = url_name[k]
        selector = get_selector(dic['url'])
        if selector['status'] == 0:
            continue
        selector = selector['selector']
        # 基本信息
        pairs = selector.xpath('//*[@class="zlxxul"]/li/text()')
        external = selector.xpath('//*[@class="zlxxul"]/li/a/text()')
        length = len(external)
        if length != 0:
            for line in pairs[:-length]:
                line = line.strip()
                if "：" in line:
                    p,v = line.split("：")
                    dic_local[p] = v
                elif ":" in line:
                    p, v = line.split(":")
                    dic_local[p] = v
            i = 0
            for k in pairs[-length:]:
                p = k.strip().strip(":").strip("：")
                dic_local[p] = external[i]
                i += 1
        # 代表作品
        try:
            k = selector.xpath('//*[@class="zldbz"]/span/text()')[0].strip("：").strip(":")
            v = selector.xpath('//*[@class="zldbz"]/text()')[0]
            dic_local[k] = v
        except:
            pass

        # 简介
        try:
            info = selector.xpath('//*[@class="zlnr-s1"]/text()')
            if len(info) == 0:
                info = selector.xpath('//*[@class="zlnr-s1"]/p/text()')
            dic_local['intro'] = info[0]
        except:
            pass
        dic[k] = dic_local
    with open(save_path,'w',encoding='utf8') as f:
        json.dump(dic,f)



if __name__=="__main__":
    # get_all_category()
    get_detail()