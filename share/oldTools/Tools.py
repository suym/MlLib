#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Tools.py
有用的工具包
'''

__author__ = "Su Yumo <suym@buaa.edu.cn>"

import operator
import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


def Merge_data_split_window(dataset,set_num=10):
    '''
    将单条命令行按照时间窗口合并成多条，减小统计误差
    '''
    new_dataset = []
    nums = len(dataset)
    count = 0
    tem = []
    for data in dataset:
        tem.extend(data)
        count = count +1
        if count % set_num == 0:
            new_dataset.append(tem)
            tem = []
    #将最后没整除set_num的数据归为一条
    new_dataset.append(tem)
    print "Number of the dataset: %s" % nums
    print "Number of the new_dataset: %s" % len(new_dataset)
    print "Number of the set_num: %s" % set_num
    
    return new_dataset

def Merge_data_sliding_window(dataset,set_num=10):
    '''
    将单条命令行按照时间滑动窗口合并成多条，减小统计误差
    '''
    new_dataset = []
    nums = len(dataset)
    count = 0
    new_count = 0
    tem = []
    while count != nums:#要检查一下是否是nums-1
        tem.extend(dataset[count])
        count = count +1
        new_count = new_count +1
        if new_count % set_num == 0:
            new_dataset.append(tem)
            tem = []
            count = count - set_num +1
            new_count = 0
    print "Number of the dataset: %s" % nums
    print "Number of the new_dataset: %s" % len(new_dataset)
    print "Number of the set_num: %s" % set_num
    
    return new_dataset

def Store_data(dir_of_inputdata,dataset):
    '''
    将数据样本写入文本
    dir_of_inputdata：存储文本的文件
    dataset：输入的文本数据
    '''
    with open(dir_of_inputdata,'w+') as f:
        for dataline in dataset:
            f.write(str(dataline)+'\n')

def Mark_data(dataset):
    '''
    将样本数据，按照是否存在'sudo'，分为两类并标记
    dataset：样本数据集
    '''
    classset = [0]*len(dataset)
    i = 0
    for document in dataset:
        if 'sudo' in document: #有可能会出现sudo在'sudo****'，被认为存在sudo的问题
            classset[i] = 1
        i = i + 1
        
    return classset

def CreateVocabList(dataset):
    '''
    利用所有的样本数据，生成对应的词汇库
    dataset：样本数据集
    '''
    vocabset = set([])
    for document in dataset:
        vocabset = vocabset | set([str(i) for i in document])
    #去掉sudo这个特殊的字符串
    #vocabset.remove('sudo')
    print 'The length of the vocabulary: %s' %len(vocabset)
    
    return list(vocabset)

def SetofWords2Vec(vocablist,inputset):
    '''
    利用词汇库，将文本数据样本，转化为对应的词条向量
    vocablist：词汇表
    inputset：文本数据集
    '''
    datavec = []
    for document in inputset:
        tem = [0]*len(vocablist)
        for word in document:
            if str(word) in vocablist:
                tem[vocablist.index(str(word))] = 1
            else:
                #print "the word : %s is not in my vocabulary!" % word
                pass
        if sum(tem) > 0:
            datavec.append(tem)
        
    return datavec

def BagofWords2Vec(vocablist,inputset):
    '''
    利用词汇库，将文本数据样本，按照词频转化为对应的词频向量
    vocablist：词汇表
    inputset：文本数据集
    '''
    datavec = []
    for document in inputset:
        tem = [0]*len(vocablist)
        for word in document:
            if str(word) in vocablist:
                tem[vocablist.index(str(word))] += 1
            else:
                #print "the word : %s is not in my vocabulary!" % word
                pass
        if sum(tem) > 0:
            datavec.append(tem)
        
    return datavec

def PolyofWords2Vec(vocablist,inputset):
    '''
    利用词汇库，将文本数据样本，转化为对应的词条向量
    vocablist：词汇表
    inputset：文本数据集
    '''
    datavec = []
    for document in inputset:
        tem = [0]*len(document)
        for word in document:
            if str(word) in vocablist:
                tem[document.index(str(word))]=vocablist.index(str(word))
            else:
                pass
        if sum(tem) > 0:
            datavec.append(tem)
            
    return datavec

def Load_data(datavec,classset):
    '''
    将数据样本划分为训练集和测试集
    参数stratify=Y，分层抽样，保证测试数据的无偏性
    '''
    X = datavec
    Y = classset
    print "Number of the positive class: %s" % classset.count(1)
    print "Number of the negative class: %s" % classset.count(0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.75, test_size=0.25,stratify=Y,random_state=0)
    
    return X_train, X_test, y_train, y_test

def CalcMostFreq(vocabList,dataset):
    '''
    返回前30个高频词
    vocabList：词汇表
    fullText：文本数据样本
    '''
    freqDict = {}
    fullText = []
    new_vocablist = []
    for data in dataset:
        fullText.extend(data)
        
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    sortedFreq = sortedFreq[:30]
    
    for pairw in sortedFreq:
        #pairw[0]是字符串，不是list，所以用append，而不是extend
        new_vocablist.append(pairw[0])
        
    return new_vocablist

def CalLessFreq(vocabList,dataset):
    '''
    去掉词汇库中在文本数据中只出现一次的词
    '''
    freqDict = {}
    fullText = []
    new_vocablist = []
    for data in dataset:
        fullText.extend(data)
        
    for token in vocabList:
        if fullText.count(token) > 1:
            freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    print 'The length of the vocabulary: %s after removed the low-frequency words'%len(sortedFreq)
    
    for pairw in sortedFreq:
        #pairw[0]是字符串，不是list，所以用append，而不是extend
        new_vocablist.append(pairw[0])
        
    return new_vocablist
    
def Modify_counts_with_TFIDF(dataset):
    '''
    使用TFIDF对文档词频矩阵进行加权调整，其中稀少的词的的权重会大于一般性的单词
    使用详情：https://blog.csdn.net/zhzhji440/article/details/47193731
    '''
    X = np.array(dataset)
    #对行求和，得到出现在文章i中所有索引词出现的次数
    WordsPerDoc = np.sum(X, axis=1)
    #语料库文章出现索引词j的文章数，也就是源矩阵中j列中非零元素的个数，其中参数'i'代表整形int，
    DocsPerWord = np.sum(np.asarray(X > 0, 'i'), axis=0)
    rows, cols = X.shape
    new_dataset = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            new_dataset[i,j] = (float(X[i,j]) / WordsPerDoc[i]) * np.log(float(rows) / DocsPerWord[j])
    
    return new_dataset.tolist()
    
def Data_process(dataset,options='minmaxscaler'):
    '''
    对数据进行z-score标准化或者标准化到0到1
    '''
    if options == 'z-score':
        x_ori = np.array(dataset)
        scaler = StandardScaler()
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    if options == 'minmaxscaler':
        x_ori = np.array(dataset)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    
    return X.tolist(),scaler

def Data_inverse_transform(dataset,scaler):
    x_ori = np.array(dataset)
    X = scaler.inverse_transform(x_ori)
    
    return X.tolist()
    
def TrainNB(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数(此处仅处理两类分类问题)
    trainMatrix:训练集文档矩阵
    trainCategory:训练集每篇文档类别标签
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pPositive = sum(trainCategory)/float(numTrainDocs)
    #初始化所有词出现数为1，并将分母初始化为2，避免某一个概率值为0
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0 
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) #这里sum(trainMatrix[i])对伯努利贝叶斯方法可能有问题，对词袋模型没问题
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #将结果取自然对数，避免下溢出，即太多很小的数相乘造成的影响
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    
    return p0Vect,p1Vect,pPositive

def TestNB(testMatrix,testCategory,p0Vec,p1Vec,pClass1):
    '''
    朴素贝叶斯分类器测试函数(此处仅处理两类分类问题)
    testMatrix:测试集文档矩阵
    testCategory:测试集每篇文档类别标签
    p0Vec, p1Vec, pClass1:分别对应TrainNB计算得到的3个概率
    '''
    errorCount = 0
    numOfTestSet = len(testMatrix)
    for index in range(numOfTestSet):
        if ClassifyNB(testMatrix[index],p0Vec,p1Vec,pClass1) != testCategory[index]:
            errorCount += 1
    errorRate = float(errorCount)/numOfTestSet
    
    return errorRate
    
def ClassifyNB(testVec,p0Vec,p1Vec,pClass1):
    '''
    分类函数
    testVec:要分类的向量
    p0Vec, p1Vec, pClass1:分别对应TrainNB计算得到的3个概率
    '''
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def Cal_main_labels(labels):
    '''
    计算labels中主要集中在那些类别中
    '''
    freqDict = {}
    for clst_lab in set(labels):
        if clst_lab != -1:
            freqDict['class %s:'%clst_lab]=list(labels).count(clst_lab)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedFreq
    
def Show_data(vocabset,dataset,labels,options='one'):
    '''
    在已经标签的数据中，将每个大于0的词输出来
    vocabset：词汇表
    dataset：词频向量数据集
    labels：词频向量的标签
    options：是一种选项
    '''
    freq_data = Cal_main_labels(labels)
    
    show_data = []
    num_vocabset = len(vocabset)
    num_dataset = len(dataset)
    
    if options == 'one':
        for data_index in range(num_dataset):
            tem = []
            tem_dict = {}
            for word_index in range(num_vocabset):
                if dataset[data_index][word_index] > 0.1:
                    tem.extend([vocabset[word_index]])
            tem_dict['class %s:'%labels[data_index]] = tem
            show_data.append(tem_dict)
            
    if options == 'two':
        '''
        用于特征向量的维度是一条文本的个数，而不是词典的长度
        '''
        for data_index in range(num_dataset):
            tem = []
            tem_dict = {}
            for word_index in range(len(dataset[data_index])):
                vocabset_index = dataset[data_index][word_index]
                if vocabset_index > 0.1:
                    tem.extend([vocabset[vocabset_index]])
            tem_dict['class %s:'%labels[data_index]] = tem
            show_data.append(tem_dict)
            
    if options == 'three':
        '''
        用于特征向量的维度是一条文本的个数，而不是词典的长度
        '''
        for data_index in range(num_dataset):
            tem_dict = {}
            tem_dict['class %s:'%labels[data_index]] = dataset[data_index]
            show_data.append(tem_dict)
    
    for f_data in freq_data:
        for s_data in show_data:
            if [f_data[0]] == s_data.keys():
                #这里的s_data.values()[0]的[0]是为了去掉一个list括号.即[[]]
                print "%s %s" %(f_data,s_data.values()[0])
                break
            
    return show_data

def Find_exception(dir_of_inputdata,dataset):
    '''
    寻找异常类-1
    dir_of_inputdata：输入目录
    dataset：数据
    '''
    with open(dir_of_inputdata,'w+') as f:
        #f.write("this is the beginning.\n")
        for dataline in dataset:
            if 'class -1' in str(dataline):
                #if 'my_NaN' in str(dataline):
                f.write(str(dataline)+'\n')

def Find_exception_reason(dataset,dataset_s,clst_labels):
    '''
    找出异常用户的原因
    '''
    num_dataset = len(dataset_s)
    show_data = []
    
    data_tem = pd.DataFrame(dataset)
    finalTable=pd.merge(data_tem,pd.DataFrame(clst_labels,columns=["labels"]),how="left",left_index=True,right_index=True)
    
    dataset = np.array(dataset)
    #正常人的平均行为
    nomalTable=finalTable[finalTable["labels"]!=-1].agg(["mean"]).round(4)
    #去掉label的列
    nomalTable = nomalTable.drop(['labels'],axis =1)
        
    for data_index in range(num_dataset):
        tem_dict = {}
        #nomalTable.values[0]的[0]是为了去掉[[]]的一层括号,把每一个维度和正常的均值点偏差求和，并保留两位小数
        user_score =  round(abs((dataset[data_index]- nomalTable.values[0])).sum(),2)
        tem_dict['class %s, user_score %s, '%(clst_labels[data_index],user_score)] = dataset_s[data_index]
        show_data.append(tem_dict)
    
    return show_data
    
def Add_exception_scores(dataset_s,clst_labels,scores_pred):
    '''
    保存模型中已经计算出来的异常度
    clst_labels：数据类别标签
    scores_pred：异常得分
    '''
    num_dataset = len(dataset_s)
    show_data = []
        
    for data_index in range(num_dataset):
        tem_dict = {}
        tem_dict['class %s, user_score %s, '%(clst_labels[data_index],scores_pred[data_index])] = dataset_s[data_index]
        show_data.append(tem_dict)
    
    return show_data
                    
def Duration(seconds):
    '''
    将传入的秒，转化为其他计时单位
    '''
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

def Visualize(dataset,c_label='b',dims=2):
    '''
    可视化数据
    dataset:数据样本
    c_label:类别颜色
    dims:可视化维度
    '''
    if dims == 2:
        x = dataset[:,0]
        y = dataset[:,1]

        ax = plt.subplot(111)
        ax.scatter(x,y,c=c_label,label="t-SNE")

    elif dims == 3:
        x = dataset[:,0]
        y = dataset[:,1]
        z = dataset[:,2]

        ax = plt.subplot(111, projection='3d')
        ax.scatter(x,y,z,c=c_label,label="t-SNE")

    plt.show()

    

