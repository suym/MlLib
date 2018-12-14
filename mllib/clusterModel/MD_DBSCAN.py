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


def main_model(dir_of_dict):
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    model_name = 'MD_DBSCAN'
    #dir_of_dict = sys.argv[1]
    bag = too.Read_info(dir_of_dict,'non-supervision')
    name_dict,task_id,job_id,train_result_dir,\
    names_str,names_num,names_show,\
    dir_of_inputdata,dir_of_outputdata,open_pca,normalized_type = bag
    dir_of_storePara = train_result_dir + '/%s_Parameters.json'%(str(task_id)+'_'+str(job_id)+'_'+model_name)

    DBSCAN_options = name_dict['DBSCAN_options']

    column_names = names_str + names_num
    column_names_show = names_str + names_num + names_show

    time_start = time()
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
    X = too.Data_process(X_datavec,normalized_type)

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
    best_epsilon,best_num = mlp.GS_DBSCAN_parameter(X,DBSCAN_options)
    clst_labels,evaluate_score = mlp.Model_DBSCAN(X,best_epsilon,best_num,DBSCAN_options)
    #寻找异常原因，并给出异常值和基准值
    exception_data,base_value = too.Find_exception_reason(X,dataset_show,clst_labels,column_names)
    #存储异常结果
    exception_data.to_csv(dir_of_outputdata,index=False)

    duration = too.Duration(time()-time_start)
    print 'Total run time: %s'%duration

if __name__ == "__main__":
    main_model(dir_of_dict)

