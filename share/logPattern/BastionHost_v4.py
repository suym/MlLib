#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
BastionHost.py
用于堡垒机词频数据的算法模型
'''

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"


import numpy as np
import pandas as pd
import os
import sys
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics


def Data_process(dataset,options='minmaxscaler'):
    '''
    对数据进行z-score标准化或者标准化到0到1
    dataset：输入数据，要求类型为多维list，行是样本数，列是特征值
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
    
    return X.tolist()


def Gs_DBSCAN_parameter(dataset):
    '''
    利用贪心算法（坐标下降算法），寻找最优的epsilon和min_samples参数
    dataset：输入数据，要求类型为多维list，行是样本数，列是特征值
    '''
    X = dataset
    X = X[0:5000]
    epsilons = [0.001,0.05,0.06,0.08,0.1,0.2]
    min_samples = [2,3,4,5,10]
    evalue = []
    mvalue = []
    for epsilon in epsilons:
        clst = DBSCAN(eps = epsilon, n_jobs = 20)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            #轮廓系数评价指标
            evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            #备用聚类指标，计算这个指标，速度快
            #evalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_epsilon = epsilons[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Epsilon Value: %s" % epsilons
    print "============================================="
    for num in min_samples:
        clst = DBSCAN(eps = best_epsilon,min_samples = num,n_jobs = 20)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            #轮廓系数评价指标
            mvalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            #备用聚类指标，计算这个指标，速度快
            #mvalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            mvalue.append(-1)#为了后面的mvalue.index(max(mvalue))可以找到正确的mindex而补了一个-1的位置
    if len(mvalue) == mvalue.count(-1):
        raise NameError('Empty Sequence')
    mindex = mvalue.index(max(mvalue))
    best_num = min_samples[mindex]
    print "Evaluate Ratio: %s" % mvalue
    print "Min Samples Value: %s" % min_samples
    print "============================================="
    print "Best Epsilon: %s" % best_epsilon
    print "Best Min Samples: %s" % best_num
    
    return best_epsilon,best_num

def Model_DBSCAN(dataset,best_epsilon=0.1,best_num=5):
    '''
    使用DBSCAN聚类结果为数据贴标签
    dataset：输入数据，要求类型为多维list，行是样本数，列是特征值
    '''
    X = dataset
    
    clst = DBSCAN(eps = best_epsilon, min_samples = best_num, n_jobs = 20)
    clst.fit(X)
    clst_labels = clst.labels_
    if len(set(clst_labels))>1:
        #轮廓系数评价指标
        evalue=metrics.silhouette_score(X,clst.labels_,metric='euclidean')
        #备用聚类指标，计算这个指标，速度快
        #evalue=metrics.calinski_harabaz_score(X,clst.labels_)
        #用于测试
        #evalue=1
    else:
        #小于0就行，-100仅仅是为了方便
        evalue=-100
    #输出评价系数
    print "============================================="
    print "Evaluate Ratio: %s" % evalue
    print "============================================="
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    
    return clst_labels,evalue

def Find_exception_reason(dataset,dataset_s,clst_labels,column_names):
    '''
    找出异常用户的原因，并给出异常值
    dataset：输入数据，要求类型为多维list，行是样本数，列是特征值
    dataset_s：输入数据，要求类型为pandas.DataFrame，行是样本数，列是特征值，比dataset多了用户和日期列，用于显示异常数据
    clst_labels：数据的聚类标签，要求类型为list
    column_names:dataset的列名
    '''
    num_dataset = len(dataset)
    #深度复制，避免修改show_data会影响到dataset_s
    show_data = copy.deepcopy(dataset_s)
    for_base_value = 
    
    data_tem = pd.DataFrame(dataset,columns=column_names)
    finalTable=pd.merge(data_tem,pd.DataFrame(clst_labels,columns=["labels"]),how="left",left_index=True,right_index=True)
    
    dataset = np.array(dataset)
    #正常人的平均行为
    nomalTable=finalTable[finalTable["labels"]!=-1].agg(["mean"]).round(4)
    #去掉label的列
    nomalTable = nomalTable.drop(['labels'],axis =1)
        
    for data_index in range(num_dataset):
        #nomalTable.values[0]的[0]是为了去掉[[]]的一层括号,把每一个维度和正常的均值点偏差求和，并保留两位小数
        user_score =  round(abs((dataset[data_index]- nomalTable.values[0])).sum(),2)
        show_data.loc[data_index,u'操作异常度'] = user_score
        show_data.loc[data_index,u'异常类别'] = 'class'+str(clst_labels[data_index])

    #只获取异常数据
    exceptionTable = show_data[show_data[u'异常类别']=='class-1']
        
    return exceptionTable,nomalTable
    
def main():
    #输入文件目录
    input_dir = sys.argv[1]
    #输出文件目录
    output_dir = sys.argv[2]
    #用于去掉一些不要的特征列
    names = sys.argv[3]
    #得到文件夹下的所有子目录和文件名称
    files = os.listdir(input_dir)
    for subdir in files:
        subdir = input_dir + '/' + subdir
        if os.path.isdir(subdir): #判断是否是文件夹，是文件夹才打开
            file = os.listdir(subdir)
            for subfile in file:
                #排除隐藏文件
                if subfile[0]!='.':
                    #定义输入和输出文件名
                    input_dir_file = input_dir + '/' + subdir + '/' + subfile
                    output_dir_file = output_dir + '/' + subdir + '.log'
                    base_value_file = output_dir + '/' + subdir + 'base_value'+'.log'

                    dataset = pd.read_csv(input_dir_file,encoding='gbk')
                    #用于测试，跑少量数据，可注释掉
                    dataset = dataset[0:1000]
                    #dataset_s用于显示异常结果
                    dataset_s = dataset
                    #dataset作为算法的输入数据
                    dataset = dataset.drop(names, axis = 1)
                    #获得dataset的列名
                    column_names = dataset.columns
                    dataset = dataset.values.tolist()
                    #数据预处理，特征归一化
                    datavec = Data_process(dataset)
                    try :
                        #DBSCAN算法参数优化
                        best_epsilon,best_num = Gs_DBSCAN_parameter(datavec)
                        #运行DBSCAN算法
                        clst_labels,evalue_score = Model_DBSCAN(datavec,best_epsilon,best_num)
                        if (evalue_score > 0.6) & (-1 in set(clst_labels)):
                            #寻找异常原因，并给出异常值和基准值
                            exception_data,base_value= Find_exception_reason(datavec,dataset_s,clst_labels,column_names)
                            #储存异常结果
                            exception_data.to_csv(output_dir_file,index=False,encoding='gbk')
                            #储存基准值
                            base_value.to_csv(base_value_file,index=False,encoding='gbk')
                    except NameError:
                        pass
        
if __name__ == "__main__":
    main()

    
