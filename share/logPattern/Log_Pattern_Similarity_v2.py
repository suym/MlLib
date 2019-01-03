#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Log_Pattern_Similarity_v2.py
用于计算日志pattern之间的相似度
'''

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import re
import json
import pandas as pd
from time import time
import numpy as np
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

class My_exception(Exception):  # 继承 Exception
    def __init__(self, msg):
        self.message = msg
 
    def __str__(self):  # 被print调用时执行，可以不写
        return self.message

def Duration(seconds):
    seconds = long(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365.242199)

    minutes = long(minutes)
    hours = long(hours)
    days = long(days)
    years = long(years)

    duration = []
    if years > 0:
        duration.append('%d year' % years + 's'*(years != 1))
    else:
        if days > 0:
            duration.append('%d day' % days + 's'*(days != 1))
        if hours > 0:
            duration.append('%d hour' % hours + 's'*(hours != 1))
        if minutes > 0:
            duration.append('%d minute' % minutes + 's'*(minutes != 1))
        if seconds > 0:
            duration.append('%d second' % seconds + 's'*(seconds != 1))

    return ' '.join(duration)

def Parse_data(dir_of_inputdata):
    X_dataset = []
    Y_dataset = []
    X_tem = []
    X_ori = {}
    maxlen = 0
    regEx = re.compile('\\s*')
    with open(dir_of_inputdata) as f:
        for line in f.readlines():
            data = json.loads(line)
            for data_str in data:
                #获得raw_event,PID
                X_tem.append(data_str['value'])
                Y_dataset.append(str(data_str['key']))
                X_ori[str(data_str['key'])] = data_str['value']

    for line in X_tem:
        #将句子分词
        listoftoken = regEx.split(line)
        #去掉空格
        tem = [tok for tok in listoftoken if len(tok)>0]
        #获得最长句子的单词数
        if len(tem)>maxlen:
            maxlen = len(tem)
        #将一个list中的字符拼接成字符串，不再是list
        tem_str = ' '.join(tem)
        X_dataset.append(tem_str)

    print "Max_Sequences_Length: %s"%maxlen
    print'----------------------------------------------'

    if len(X_dataset)!=len(Y_dataset):
        raise My_exception("len(x)!=len(y)--1")

    return X_dataset,Y_dataset,maxlen,X_ori

def Data_process(X_dataset,maxlen,window_size):
    #得到所有word的词频，X_dataset形式为['a s d','as df sd']
    word_freqs = Counter()
    for words in X_dataset:
        for word in words.split():
            word_freqs[word] = word_freqs[word] +1 
    #得到word to index，i从0开始
    word_index = {x: i+1 for i, x in enumerate(word_freqs)}
    print "The length of the vocabulary: %s"%(len(word_index)+1)
    print'----------------------------------------------'
    #将所有的词编码为数字
    x_data = []
    for words in X_dataset:
        x_tem = []
        for word in words.split():
            if word in word_index:
                x_tem.append(word_index[word])
            else:
               print "the word:%s is not in vocabulary"%word
        #不足最大长度的，在末尾补0
        if len(x_tem) < maxlen:
            x_tem = x_tem +[0]*(maxlen-len(x_tem))
        x_data.append(x_tem)
    
    x_onehot_data = []
    #将数字编码为独热码
    for words in x_data:
        #创造滑动间隔数据,new_words形如[['a','b','c'],['b','c','d']]
        new_words = Merge_data_sliding_window(words,window_size)
        x_tem = []
        for n_word in new_words:
            tem = [0]*(len(word_index)+1)
            for word in n_word:               
                tem[word] = 1
            x_tem.extend(tem) 
        x_onehot_data.append(x_tem)

    return x_onehot_data

def Merge_data_sliding_window(dataset,set_num=10):
    '''
    将单个词按照时间滑动窗口合并成多条
    '''
    new_dataset = []
    nums = len(dataset)
    count = 0
    new_count = 0
    tem = []
    while count != nums:#要检查一下是否是nums-1
        tem.append(dataset[count])
        count = count +1
        new_count = new_count +1
        if new_count % set_num == 0:
            new_dataset.append(tem)
            tem = []
            count = count - set_num +1
            new_count = 0

    return new_dataset


def Cal_data_sim(X_onehot_data):
    results = cosine_similarity(X_onehot_data)
    #只取出矩阵上三角
    results = np.triu(results,0)

    return results

def Save_data(cos_sim_results,Y_dataset,dir_of_outputdata,threshold_value,X_ori):
    if len(cos_sim_results)!=len(Y_dataset):
        raise My_exception("len(x)!=len(y)--2")
    event_num = len(Y_dataset)
    PID_1 = []
    PID_2 = []
    data = pd.DataFrame()
    for i in range(event_num):
        for j in range(event_num):
            if (cos_sim_results[i][j]>threshold_value)&(i!=j):
                PID_1.append(Y_dataset[i])
                PID_2.append(Y_dataset[j])

    data['PID_1'] = PID_1
    data['PID_2'] = PID_2
    merge_data = Merge_PID(data)
    #取出未被聚合的PID
    merge_data_flat = set()
    for data_flat in merge_data:
        merge_data_flat = merge_data_flat | set(data_flat)
    PID_3 = set(X_ori) - merge_data_flat
    #将[[1,2,3],[4,5],[6,7]]形式的merge_data转为dict
    key_value_list = {}
    ip = 1
    for m_d in merge_data:
        key_value_tem = []
        for m_d_pid in m_d:
            m_d_dic = {}
            m_d_dic[m_d_pid] = X_ori[m_d_pid]
            key_value_tem.append(m_d_dic)
        key_value_list['class %s'%ip] = key_value_tem
        ip = ip + 1 
    #将未被聚合的{1,2,3,4}形式的PID_3加入key_value_list的字典中
    pid3_tem = []
    for pid3 in PID_3:
        pid3_dic = {}
        pid3_dic[pid3] = X_ori[pid3]
        pid3_tem.append(pid3_dic)
    key_value_list['class %s'%0] = pid3_tem

    print "Number of samples: %s"%len(X_ori)
    print'----------------------------------------------'
    print "Number of categories: %s"%ip
    print'----------------------------------------------'
    print "Number of merged samples: %s"%len(merge_data_flat)
    print'----------------------------------------------'

    with open(dir_of_outputdata,'w') as f :
        json.dump(key_value_list,f)

def Merge_PID(data):
    #按照PID_1分组
    datagb = data.groupby("PID_1")
    set_PID_1 = set(data["PID_1"])
    merge_data = []
    for pid in set_PID_1:
        data_tem = datagb.get_group(pid)
        data_tem_PID_2 = data_tem["PID_2"].tolist()
        data_tem_PID_2.append(pid)
        #set的目的是为了避免出现['a','a']的现象，保证合并列表中有相同元素的列表的模块的正常运行
        merge_data.append(list(set(data_tem_PID_2)))
    #合并列表中有相同元素的列表
    num_of_PID = len(merge_data)
    for i in range(num_of_PID):
        for j in range(num_of_PID):
            x = list(set(merge_data[i]+merge_data[j]))
            y = len(merge_data[j])+len(merge_data[i])
            if i == j :
                break
            if len(x) < y:
                merge_data[i] = x
                merge_data[j] = ['UN-K']

    merge_data = [i for i in merge_data if i != ['UN-K']]

    return merge_data

def main():
    time_start = time()
    dir_of_inputdata = "log.dat"
    dir_of_outputdata = 'out_6.log'
    threshold_value = 0.9
    window_size = 10

    X_dataset,Y_dataset,maxlen,X_ori = Parse_data(dir_of_inputdata)
    X_onehot_data = Data_process(X_dataset,maxlen,window_size) 
    
    cos_sim_results = Cal_data_sim(X_onehot_data)
    Save_data(cos_sim_results,Y_dataset,dir_of_outputdata,threshold_value,X_ori)

    duration = Duration(time()-time_start)
    print 'Total run time: %s'%duration


if __name__ == '__main__':
    main()


