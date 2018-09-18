#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
ML_Package.py
机器学习的工具包
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN,KMeans
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def Gs_PCA(dataset):
    '''
    搜索最优PCA降维参数
    dataset：数据样本
    '''
    X = dataset
    num0 = 0.999
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = PCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
        if sum_t <= num0:
            ret['99.9%'] = count
    return pca.n_components_,ret

def Model_PCA(dataset,nums_component):
    '''
    将冗余自由度的数据样本进行降维
    dataset：数据样本
    nums_component：PCA的降维参数
    '''
    X = dataset
    pca = PCA(n_components=nums_component)
    pca.fit(X)
    X_r = pca.transform(X)
    
    return X_r

def Model_TSNE(dataset,nums_component=2):
    '''
    将冗余自由度的数据样本进行降维可视化
    dataset：数据样本
    nums_component：降维参数
    '''
    X = dataset
    X_tsne = TSNE(n_components=nums_component,random_state=33).fit_transform(X)
    
    return X_tsne
    
def Gs_DBSCAN_parameter(dataset):
    '''
    利用贪心算法（坐标下降算法），寻找最优的epsilon和min_samples参数
    dataset：数据样本
    '''
    X = dataset
    #epsilons = [0.001,0.05,0.06,0.07,0.08,0.1,0.2,0.3,0.4,0.5,0.9,1,2,3,5]
    epsilons = [0.001,0.05,0.1,0.2,0.5,0.7,1,3,5]
    #epsilons = [0.001,0.05,0.1,0.3]
    #min_sample = [1,2,3,4,5,10,15,20,30,50,70,80,100]
    min_sample = [1,2,3,5]
    evalue = []
    mvalue = []
    for epsilon in epsilons:
        clst = DBSCAN(eps = epsilon)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            #evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            evalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_epsilon = epsilons[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Epsilon Value: %s" % epsilons
    print "============================================="
    for num in min_sample:
        clst = DBSCAN(eps = best_epsilon,min_samples = num)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            #mvalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            mvalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            mvalue.append(-1)#为了后面的mvalue.index(max(mvalue))可以找到正确的mindex而补了一个-1的位置
    if len(mvalue) == mvalue.count(-1):
        raise NameError('Empty Sequence')
    mindex = mvalue.index(max(mvalue))
    best_num = min_sample[mindex]
    print "Evaluate Ratio: %s" % mvalue
    print "Min Samples Value: %s" % min_sample
    print "============================================="
    print "Best Epsilon: %s" % best_epsilon
    print "Best Min Samples: %s" % best_num
    
    return best_epsilon,best_num

def Model_DBSCAN(dataset,best_epsilon=0.1,best_num=5):
    '''
    使用DBSCAN聚类结果为数据贴标签
    dataset:数据样本
    '''
    X = dataset
    
    clst = DBSCAN(eps = best_epsilon, min_samples = best_num)
    clst.fit(X)
    clst_labels = clst.labels_
    if len(set(clst_labels))>1:
        evalue=metrics.silhouette_score(X,clst.labels_,metric='euclidean')
        #evalue=metrics.calinski_harabaz_score(X,clst.labels_)
        #evalue=1
    else:
        evalue=-10
    #输出评价系数
    print "============================================="
    print "Evaluate Ratio: %s" % evalue
    print "============================================="
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    
    return clst_labels,evalue

def Gs_IsolationForest_parameter(dataset):
    '''
    寻找最优的contamination参数
    dataset：数据样本
    '''
    X = dataset
    estimators = [-0.01,-0.05,-0.08,-0.1,-0.12,-0.15,-0.17,-0.2,-0.22,-0.25,-0.27,-0.3,-0.35,-0.37,-0.4]
    evalue = []
    
    data_shape = len(X)
    
    clst = IsolationForest(n_estimators = 100, n_jobs = 1, random_state = 0)
    clst.fit(X)
    scores_pred = clst.decision_function(X)
    #选出异常度最小的值   
    for estimator in estimators:
        #只要evalue中有100了，就说明目前的异常度已经最小，减小重复计算
        if 100 not in evalue:
            contamination_ratio = round(len([i for i in scores_pred if i<estimator])/data_shape,6)
        else :
            contamination_ratio = -1
        if contamination_ratio > 0:
            evalue.append(contamination_ratio)
        else :
            evalue.append(100)
    if len(evalue) == evalue.count(100):
        raise NameError('Empty Sequence')
    eindex = evalue.index(min(evalue))
    best_estimator = estimators[eindex]
    best_contamination = min(evalue)
    print "Contamination Ratio: %s" % evalue
    print "Estimator Value: %s" % [abs(i)+0.5 for i in estimators]
    print "============================================="
    print "Best Estimator: %s" % (abs(best_estimator)+0.5)
    print "Best Contamination: %s" % best_contamination
    print "============================================="
    
    return (abs(best_estimator)+0.5),best_contamination

def Model_IsolationForest(dataset,best_contamination=0.01):
    '''
    使用Isolation Forest聚类结果为数据贴标签
    dataset:数据样本
    '''
    X = dataset
    
    clst = IsolationForest(n_estimators = 100, contamination = best_contamination, n_jobs = 1, random_state = 0)
    clst.fit(X)
    
    scores_pred = clst.decision_function(X)
    clst_labels = clst.predict(X)
    new_scores = [round(abs(i)+0.5,4) for i in scores_pred]
        
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="
    
    return clst_labels,new_scores
  
def Gs_LocalOutlierFactor_parameter(dataset):
    '''
    寻找最优的neighbors和contamination参数
    dataset：数据样本
    '''
    X = dataset
    data_shape = len(X)
    neighbors = [15,20,25,30,35,40]
    estimators = [-1,-5,-10,-50,-100,-150,-200,-300,-500,-1000,-10000]
    evalue = []
    mvalue = []
    for neighbor in neighbors:
        clst = LocalOutlierFactor(n_neighbors=neighbor)
        clst.fit(X)
        neighbor_scores= clst.negative_outlier_factor_
        neighbor_contamination_ratio = round(len([i for i in neighbor_scores if i<-1])/data_shape,6)
        if neighbor_contamination_ratio > 0:
            evalue.append(neighbor_contamination_ratio)
        else :
            evalue.append(-1)
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_neighbor = neighbors[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Neighbor Value: %s" % neighbors
    print "============================================="
    clst = LocalOutlierFactor(n_neighbors=best_neighbor)
    clst.fit(X)
    scores_pred = clst.negative_outlier_factor_
    #选出异常度最小的值   
    for estimator in estimators:
        #只要evalue中有100了，就说明目前的异常度已经最小，减小重复计算
        if 100 not in evalue:
            contamination_ratio = round(len([i for i in scores_pred if i<estimator])/data_shape,6)
        else :
            contamination_ratio = -1
        if contamination_ratio > 0:
            mvalue.append(contamination_ratio)
        else :
            mvalue.append(100)
    if len(mvalue) == mvalue.count(100):
        raise NameError('Empty Sequence')
    mindex = mvalue.index(min(mvalue))
    best_estimator = estimators[mindex]
    best_contamination = min(mvalue)
    print "Contamination Ratio: %s" % mvalue
    print "Estimator Value: %s" % [abs(i) for i in estimators]
    print "============================================="
    print "Best Neighbor: %s" % best_neighbor
    print "Best Estimator: %s" % abs(best_estimator)
    print "Best Contamination: %s" % best_contamination
    print "============================================="
    
    return best_neighbor,best_contamination

def Model_LocalOutlierFactor(dataset,best_neighbor=20,best_contamination=0.01):
    '''
    使用Local Outlier Factor聚类结果为数据贴标签
    dataset:数据样本
    '''
    X = dataset
    
    clst = LocalOutlierFactor(n_neighbors = best_neighbor,contamination = best_contamination)
    
    clst_labels = clst.fit_predict(X)
    scores_pred = clst.negative_outlier_factor_
    new_scores = [round(abs(i),4) for i in scores_pred]
        
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="
    
    return clst_labels,new_scores

def Gs_KMeans_parameter(dataset):
    '''
    寻找最优的n_clusters参数
    dataset：数据样本
    '''
    X = dataset
    estimators = [2,3,4,5,6,7,8,9,10,15,20]
    evalue = []
    for num in estimators:
    	clst = KMeans(n_clusters=num, init='k-means++', n_jobs=-1)
	clst.fit(X)
    	if len(set(clst.labels_))>1:
            evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            #evalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_num_cluster = estimators[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Num of Cluster: %s" % estimators
    print "============================================="

    return best_num_cluster

 def Model_KMeans(dataset,best_num_cluster=8):
    '''
    使用KMeans聚类结果为数据贴标签
    dataset:数据样本
    '''
    X = dataset
    
    clst = KMeans(n_clusters=best_num_cluster, init='k-means++', n_jobs=-1)  
    clst_labels = clst.fit_predict(X)
       
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="
    
    return clst_labels


