#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"


import sys
sys.path.append("..")
import json
import warnings
import pandas as pd
import numpy as np
from src import ML_Package as mlp
from src import Tools_Package as too
from time import time


def main():
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    model_name = 'IsolationForest'
    dir_of_dict = '../config/cluster_columns.json'
    with open(dir_of_dict,'r') as f:
        column_lines = f.read()
        name_dict = eval(column_lines)
    names_str = name_dict['names_str']
    names_num = name_dict['names_num']
    names_show = name_dict['names_show']
    dir_of_inputdata = name_dict['dir_of_inputdata']
    dir_of_outputdata = name_dict['dir_of_outputdata']
    open_pca = name_dict['open_pca']

    column_names = names_str + names_num
    column_names_show = names_str + names_num + names_show

    time_start = time()
    dir_of_storePara = '../cluster_parameter/%sParameters.json'%model_name
    #获取数据
    dataset = pd.read_csv(dir_of_inputdata)
    #用于测试 
    #dataset = dataset[0:1000]
    
    dataset_show = dataset[column_names_show]
    
    #分别获得字符字段和数值型字段数据
    dataset_str = dataset[names_str]
    dataset_num = dataset[names_num]
    dataset_str_list = dataset_str.values.tolist()
    datavec_num_list = dataset_num.values.tolist()

    vocabset = too.CreateVocabList(dataset_str_list)
    datavec_str_list = too.BagofWords2Vec(vocabset,dataset_str_list)
    #vocabset_index = {y:i for i,y in enumerate(vocabset)}

    #将list转化为DataFrame，合并两表
    datavec_str = pd.DataFrame(datavec_str_list,columns=vocabset)
    datavec_num = pd.DataFrame(datavec_num_list,columns=names_num)
    #按照左表连接，右表可以为空
    data_tem = pd.merge(datavec_num,datavec_str,how="left",right_index=True,left_index=True)
    X_datavec = data_tem.values

    #数据归一化
    X = too.Data_process(X_datavec)

    ret_num = 'no_num'
    #PCA降维
    if open_pca == 'open_pca':
        pca_num,ret = mlp.GS_PCA(X)
        print 'PCA Information:',pca_num,ret
        print'----------------------------------------------'
        ret_num = ret['99%']
        X = mlp.Model_PCA(X,ret_num)
    #存储vocabset这个list和ret_num
    too.StorePara(dir_of_storePara,vocabset,ret_num)

    print'----------------data shape--------------------'
    print 'X.shape:',X.shape
    print'----------------------------------------------'
    evaluate_score,best_contamination = mlp.GS_IsolationForest_parameter(X)
    clst_labels,scores_pred = mlp.Model_IsolationForest(X,best_contamination)
    #添加异常原因，并给出异常值和基准值
    exception_data,base_value = too.Add_exception_reason(dataset_show,clst_labels,column_names,scores_pred)
    #存储异常结果
    exception_data.to_csv(dir_of_outputdata,index=False)

    duration = too.Duration(time()-time_start)
    print 'Total run time: %s'%duration

if __name__ == "__main__":
    main()

