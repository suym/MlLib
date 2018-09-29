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
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from time import time


def main():
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    model_name = 'RandomForestRegressor'
    dir_of_dict = sys.argv[1]
    bag = too.Read_info(dir_of_dict,'supervision')
    name_dict,options,task_id,job_id,train_result_dir,\
    names_str,names_num,names_show,Y_names,dir_of_inputdata,\
    dir_of_outputdata,open_pca,train_size,test_size,normalized_type = bag

    dir_of_storePara = train_result_dir + '/%s_Parameters.json'%(str(task_id)+'_'+str(job_id)+'_'+model_name)
    dir_of_storeModel = train_result_dir + '/%s_model.m'%(str(task_id)+'_'+str(job_id)+'_'+model_name)

    if options == 'train':
        time_start = time()
        #获取数据
        dataset = pd.read_csv(dir_of_inputdata)
        #用于测试 
        #dataset = dataset[0:1000]

        Y_datavec = dataset[Y_names].values

        #分别获得字符字段和数值型字段数据
        dataset_str = dataset[names_str]
        dataset_num = dataset[names_num]
        dataset_show = dataset[names_show]
        dataset_str_list = dataset_str.values.tolist()
        datavec_num_list = dataset_num.values.tolist()
        datavec_show_list = dataset_show.values.tolist()

        vocabset = too.CreateVocabList(dataset_str_list)
        datavec_str_list = too.BagofWords2Vec(vocabset,dataset_str_list)
        #vocabset_index = {y:i for i,y in enumerate(vocabset)}

        #将list转化为DataFrame，合并两表
        datavec_str = pd.DataFrame(datavec_str_list,columns=vocabset)
        datavec_num = pd.DataFrame(datavec_num_list,columns=names_num)
        #按照左表连接，右表可以为空
        data_tem = pd.merge(datavec_num,datavec_str,how="left",right_index=True,left_index=True)
        X_datavec = data_tem.values
        X_columns = data_tem.columns

        #数据归一化
        X_datavec = too.Data_process(X_datavec,normalized_type)
        #处理数据不平衡问题
        #X,Y =  mlp.KMeans_unbalanced(X_datavec,Y_datavec,X_columns,Y_names)
        #X,Y =  mlp.Sample_unbalanced(X_datavec,Y_datavec)
        X,Y = X_datavec, Y_datavec
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

        print'--------------Train data shape----------------'
        print 'X.shape:',X.shape
        print'----------------------------------------------'
        print 'Y.shape:',Y.shape
        print'----------------------------------------------'
        print'--------------Start %s model------------------'%model_name
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            train_size=train_size, test_size=test_size,random_state=0)
        clf_model = mlp.GS_RandomForestRegressor(X_train, X_test, y_train, y_test)
        #保存模型参数
        joblib.dump(clf_model, dir_of_storeModel)
        print'----------------------------------------------'
        too.Predict_test_data(X_test, y_test, datavec_show_list, names_show, clf_model, dir_of_outputdata,mtype='reg')
        duration = too.Duration(time()-time_start)
        print 'Total run time: %s'%duration

    if options == 'predict':
        time_start = time()
        with open(dir_of_storePara,'r') as f:
            para_dict = json.load(f)
        vocabset = para_dict['vocabset']
        ret_num = para_dict['ret_num']
        #获取数据
        X_dataset = pd.read_csv(dir_of_inputdata)

        #分别获得字符字段和数值型字段数据
        dataset_str = X_dataset[names_str]
        dataset_num = X_dataset[names_num]
        dataset_show = X_dataset[names_show]
        dataset_str_list = dataset_str.values.tolist()
        datavec_num_list = dataset_num.values.tolist()
        datavec_show_list = dataset_show.values.tolist()

        datavec_str_list = too.BagofWords2Vec(vocabset,dataset_str_list)

        #将list转化为DataFrame，合并两表
        datavec_str = pd.DataFrame(datavec_str_list,columns=vocabset)
        datavec_num = pd.DataFrame(datavec_num_list,columns=names_num)
        data_tem = pd.merge(datavec_num,datavec_str,how="left",right_index=True,left_index=True)
        X_datavec = data_tem.values

        #数据归一化
        X = too.Data_process(X_datavec,normalized_type)
        #PCA降维
        if open_pca == 'open_pca':
            X = mlp.Model_PCA(X,ret_num)

        print'-------------Pdedict data shape---------------'
        print 'X.shape:',X.shape
        print'----------------------------------------------'
        print'--------------Start %s model------------------'%model_name

        clf_model = joblib.load(dir_of_storeModel)
        too.Predict_data(X, datavec_show_list, names_show, clf_model, dir_of_outputdata,mtype='reg')
        duration = too.Duration(time()-time_start)
        print 'Total run time: %s'%duration

if __name__ == "__main__":
    main()

