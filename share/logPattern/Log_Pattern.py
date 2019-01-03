#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Log_Pattern.py
用于日志pattern合并的算法模型
'''

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import re
import os
import operator
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from time import time

from keras.models import Sequential
from keras.models import load_model 
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Dropout, Activation, Embedding, Flatten
from keras.optimizers import SGD


def make_cla_LSTM(nb_time_steps,num_labels,vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, 128, input_length=nb_time_steps))
    model.add(LSTM(64,return_sequences=True,kernel_initializer='normal',activation='relu'))
    model.add(Flatten())
    model.add(Dense(10000, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    model.add(Dense(num_labels,kernel_initializer='normal',activation='softmax'))
    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model

def GS_cla_LSTM(*data):
    if len(data)!= 6:
        raise NameError('Dimension of the input is not equal to 6')
    X_train, X_test, y_train, y_test, y_num_labels, vocab_size = data
    X_time_steps = X_train.shape[1]
    
    model = make_cla_LSTM(X_time_steps, y_num_labels, vocab_size)
    model.fit(X_train, y_train,
          epochs=2,
          batch_size=256)
    if not os.path.isfile('./my_model_v4.h5'):
        model.save('my_model_v4.h5')
    
    #model = LoadModel()

    score, acc = model.evaluate(X_test, y_test, batch_size=128)
    print "\nTest score: %.3f, accuracy: %.3f" % (score, acc)
    
    return model

def LoadModel():
    model = load_model('my_model_v4.h5')

    return model

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

def Predict_test_data(X_test, y_test, model, word_index, label_index, dir_of_outputdata):
    #下标对文本的索引表
    index_word = {v:k for k, v in word_index.items()}
    #下标对label的索引表
    index_label = {v:k for k, v in label_index.items()}
    num_of_data = 500
    with open(dir_of_outputdata,'w') as f :
        print >> f,'{}   {}      {}'.format('y_predict','y_true','X_raw_data')
        print >> f,'----------------------------------------------'
        for i in range(num_of_data):
            y_tem = model.predict(X_test[[i]])[0]
            #获得最大值的下标
            y_pred_index = np.argmax(y_tem)
            y_label_index = np.argmax(y_test[i])
            y_pred = index_label[y_pred_index]
            y_label = index_label[y_label_index]
            #x==0是pad_sequences的填充值，因此要去掉
            X_raw_data = " ".join([index_word[x] for x in X_test[i] if x!=0 ])
            print >> f,' {}      {}     {}'.format(y_pred, y_label, X_raw_data)
            print >> f,'----------------------------------------------'

def Predict_data(X,Y_dataset,id_dataset,parse_dataset, model,word_index, label_index, dir_of_outputdata):
    #下标对文本的索引表
    index_word = {v:k for k, v in word_index.items()}
    #下标对label的索引表
    index_label = {v:k for k, v in label_index.items()}
    if not(len(X)==len(Y_dataset)==len(id_dataset)==len(parse_dataset)):
        raise NameError('Dimension of the input is not equal')
    num_of_data = len(X)
    threshold_value = 0.7
    y_total = model.predict(X,batch_size=256)
    with open(dir_of_outputdata,'w') as f :
        print >> f,'{}, {}, {}, {}, {}'.format('predict_PID','old_PID','id','Parsed','predict_PID==old_PID')
        for i in range(num_of_data):
            y_tem = y_total[i]
            if max(y_tem) > threshold_value:
                #获得最大值的下标
                y_pred_index = np.argmax(y_tem)
                y_pred = index_label[y_pred_index]
            else:
                y_pred = 'new_label'
            if len(str(Y_dataset[i]))==0:
                equal_key = 'null' 
            if (str(y_pred)==str(Y_dataset[i]))&(len(str(Y_dataset[i]))>0):
                equal_key = 'true'
            if (str(y_pred)!=str(Y_dataset[i]))&(len(str(Y_dataset[i]))>0):
                equal_key = 'false'
            #X_raw_data = " ".join([index_word[x] for x in X[i] if x!=0 ])

            print >> f,'{}, {}, {}, {}, {}'.format(y_pred,Y_dataset[i],id_dataset[i],parse_dataset[i],equal_key)

def Parse_train_data(dir_of_inputdata):
    X_dataset = []
    Y_dataset = []
    X_tem = []
    maxlen = 0
    regEx = re.compile('\\W*')
    with open(dir_of_inputdata) as f:
        for line in f.readlines():
            data_str = json.loads(line)
            #获得raw_event,PID,id,Parsed
            X_tem.append(data_str['_source']['raw_event'])
            Y_dataset.append(str(data_str['_source']['PID']))

    for line in X_tem:
        #将句子分词
        listoftoken = regEx.split(line)
        #去掉空格并将字符转为小写
        tem = [tok.lower() for tok in listoftoken if len(tok)>0]
        #将句子中的数字转化为特定的字符isdigit
        tem_digit = ['isdigit' if x.isdigit() else str(x) for x in tem]
        #获得最长句子的单词数
        if len(tem_digit)>maxlen:
            maxlen = len(tem_digit)
        #将一个list中的字符拼接成字符串，不再是list
        tem_isdigit = ' '.join(tem_digit)
        X_dataset.append(tem_isdigit)

    xytable = pd.DataFrame()
    xytable['X'] = X_dataset
    xytable['Y'] = Y_dataset
    #将样本中空的字符转化为NAN
    xytable['X'] = xytable['X'].apply(lambda x: np.NaN if len(str(x))==0 else x)
    xytable['Y'] = xytable['Y'].apply(lambda x: np.NaN if len(str(x))==0 else x)
    xytable = xytable.dropna()
    #计算出数据中主要的类别
    xytable = CalcMostLabel(xytable)
    
    X_dataset = xytable['X'].values.tolist()
    Y_dataset = xytable['Y'].values.tolist()

    print "Max_Sequences_Length: %s"%maxlen
    print'----------------------------------------------'

    return X_dataset, Y_dataset, maxlen

def Parse_predict_data(dir_of_inputdata):
    X_dataset = []
    Y_dataset = []
    id_dataset = []
    parse_dataset = []
    X_tem = []
    regEx = re.compile('\\W*')
    with open(dir_of_inputdata) as f:
        for line in f.readlines():
            data_str = json.loads(line)
            #获得raw_event,id
            X_tem.append(data_str['_source']['raw_event'])
            Y_dataset.append(str(data_str['_source']['PID']))
            id_dataset.append(str(data_str['_id']))
            parse_dataset.append(str(data_str['_source']['Parsed']))
    for line in X_tem:
        #将句子分词
        listoftoken = regEx.split(line)
        #去掉空格并将字符转为小写
        tem = [tok.lower() for tok in listoftoken if len(tok)>0]
        #将句子中的数字转化为特定的字符isdigit
        tem_digit = ['isdigit' if x.isdigit() else str(x) for x in tem]
        #将一个list中的字符拼接成字符串，不再是list
        tem_isdigit = ' '.join(tem_digit)
        X_dataset.append(tem_isdigit)

    xytable = pd.DataFrame()
    xytable['X'] = X_dataset
    xytable['Y'] = Y_dataset
    xytable['id'] = id_dataset
    xytable['Parsed'] = parse_dataset
    #将样本中空的字符转化为NAN,但是保留Y中的空格
    xytable['X'] = xytable['X'].apply(lambda x: np.NaN if len(str(x))==0 else x)
    xytable['id'] = xytable['id'].apply(lambda x: np.NaN if len(str(x))==0 else x)
    xytable = xytable.dropna()
    
    X_dataset = xytable['X'].values.tolist()
    Y_dataset = xytable['Y'].values.tolist()
    id_dataset = xytable['id'].values.tolist()
    parse_dataset = xytable['Parsed'].values.tolist()

    return X_dataset,Y_dataset,id_dataset,parse_dataset

def CalcMostLabel(xytable):
    #按照类别的数量，从多到少排序
    Y_tem = sorted(Counter(xytable['Y']).items(),key=operator.itemgetter(1), reverse=True)
    low_value = 500
    hight_value = 30000
    Y_tem2 = []
    Y_tem3 = []
    new_xy = pd.DataFrame()
    #只选择样本数大于low_value的类别数据
    for i in Y_tem:
        if (i[1] > low_value)&(i[1] < hight_value):
            Y_tem2.append(i[0])
        if i[1] > hight_value:
            Y_tem3.append(i[0])
    xy_tem = xytable[xytable['Y'].isin(Y_tem2)]
    new_xy = pd.concat([new_xy,xy_tem])
    #限制多数类的数据量
    if len(Y_tem3) > 0: 
        for label in Y_tem3:
            Y_tem4 = xytable[xytable['Y']==label].sample(n=hight_value,random_state=0)
            new_xy = pd.concat([new_xy,Y_tem4])

    return new_xy

def Train_data_process(X_dataset,Y_dataset,Max_Sequences_Length):
    #得到所有word的词频，X_dataset形式为['a s d','as df sd']
    word_freqs = Counter()
    for words in X_dataset:
        for word in words.split():
            word_freqs[word] = word_freqs[word] +1 
    #得到word to index，i从2开始，word_freqs.most_common返回的是形如('',12)，因此用x[0]
    word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(len(word_freqs)-1))}
    #将所有的未知字符都归为'UNK'
    word_index["UNK"] = 1
    #将所有的词编码为数字
    x_data = []
    for words in X_dataset:
        x_tem = []
        for word in words.split():
            if word in word_index:
                x_tem.append(word_index[word])
            else:
                x_tem.append(word_index['UNK'])
        x_data.append(x_tem)
    #将长度不足Max_Sequences_Length的句子用0填充
    x_data = pad_sequences(x_data, maxlen=Max_Sequences_Length)
    #将字符串的标签转化为数字标签,从0开始
    label_index = {y:i for i,y in enumerate(Counter(Y_dataset))}
    y_index = []
    #Y_dataset形式为['asd','asdfsd']
    for words in Y_dataset:
        y_tem = []
        if words in label_index:
            y_tem.append(label_index[words])
        else:
            raise NameError('Not found label')
        y_index.extend(y_tem)
    #输出每个标签的数量
    print 'Counter:y',Counter(y_index)
    print'----------------------------------------------'
    #随机采样的少数类，解决类别不平衡问题
    ros = RandomOverSampler(random_state=0)
    x_resampled, y_resampled = ros.fit_sample(x_data, y_index)
    print 'Counter:y after using RandomOverSampler',Counter(y_resampled)
    print'----------------------------------------------'
    num_used = 20000
    if len(x_resampled)>num_used*len(Counter(y_index)):
        #每个类别分层采样num_used个事例
        x_NoUse_train, x_resampled, y_NoUse_train, y_resampled = train_test_split(x_resampled, y_resampled,
                                          train_size=None, test_size=num_used*len(Counter(y_index)),stratify=y_resampled,random_state=0)
    print 'Counter:used y',Counter(y_resampled)
    print'----------------------------------------------'
    print 'Size of word_index: %s'%len(word_index)
    print'----------------------------------------------'
    #将标签处理成one-hot向量，比如6变成了[0,0,0,0,0,0,1,0,0,0,0,0,0]
    y_label = np_utils.to_categorical(y_resampled)
    print 'word to index:'
    print word_index
    print'----------------------------------------------'
    print 'label to index:'
    print label_index
    print'----------------------------------------------'

    return x_resampled,y_label,len(word_index),word_index,label_index

def Predict_data_process(X_dataset,word_index,Max_Sequences_Length):
    #将所有的词编码为数字
    x_data = []
    for words in X_dataset:
        x_tem = []
        for word in words.split():
            if word in word_index:
                x_tem.append(word_index[word])
            else:
                x_tem.append(word_index['UNK'])
        x_data.append(x_tem)
    #将长度不足Max_Sequences_Length的句子用0填充
    x_data = pad_sequences(x_data, maxlen=Max_Sequences_Length)

    return x_data

def StorePara(dir_of_storePara,word_index,label_index,Max_Sequences_Length):
    para_dict = {}
    para_dict['word_index']=word_index
    para_dict['label_index'] = label_index
    para_dict['Max_Sequences_Length'] = Max_Sequences_Length
    with open(dir_of_storePara,'w') as f:
        json.dump(para_dict,f)

def main():
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    options = sys.argv[1]
    dir_of_inputdata = sys.argv[2]
    dir_of_outputdata = sys.argv[3]

    if options == 'train':
        time_start = time()
        dir_of_storePara = './Parameters.json'
        X_dataset, Y_dataset, Max_Sequences_Length = Parse_train_data(dir_of_inputdata)
        X,Y,vocab_size,word_index,label_index = Train_data_process(X_dataset,Y_dataset,Max_Sequences_Length)
        y_num_labels = Y.shape[1]
        print'--------------Train data shape----------------'
        print 'X.shape:',X.shape
        print'----------------------------------------------'
        print 'Y.shape:',Y.shape
        print'----------------------------------------------'
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75, test_size=0.25, random_state=0)
        model = GS_cla_LSTM(X_train, X_test, y_train, y_test, y_num_labels, vocab_size)
        Predict_test_data(X_test, y_test, model, word_index, label_index, dir_of_outputdata)
        StorePara(dir_of_storePara, word_index, label_index, Max_Sequences_Length)
        duration = Duration(time()-time_start)
        print 'Total run time: %s'%duration
        
    if options == 'predict':
        time_start = time()
        dir_of_storePara = './Parameters.json'
        with open(dir_of_storePara,'r') as f:
            para_dict = json.load(f)
        word_index = para_dict['word_index']
        label_index = para_dict['label_index']
        Max_Sequences_Length = para_dict['Max_Sequences_Length']
        X_dataset,Y_dataset,id_dataset,parse_dataset = Parse_predict_data(dir_of_inputdata)
        X = Predict_data_process(X_dataset,word_index,Max_Sequences_Length)
        print'-------------Pdedict data shape---------------'
        print 'X.shape:',X.shape
        print'----------------------------------------------'
        model = LoadModel()
        Predict_data(X,Y_dataset,id_dataset,parse_dataset, model,word_index, label_index, dir_of_outputdata)
        duration = Duration(time()-time_start)
        print 'Total run time: %s'%duration
if __name__ == "__main__":
    main()

