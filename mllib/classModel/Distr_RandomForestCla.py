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
from src import Distr_ML_Package as dmp 
from collections import Counter
from time import time

from pyspark.sql import SparkSession
from pyspark.ml.feature import RFormula,VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

def main():
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    model_name = 'Distr_RandomForestCla'
    dir_of_dict = sys.argv[1]
    bag = too.Read_info(dir_of_dict,'supervision')
    name_dict,options,task_id,job_id,train_result_dir,\
    names_str,names_num,names_show,Y_names,dir_of_inputdata,\
    dir_of_outputdata,open_pca,train_size,test_size,normalized_type = bag

    dir_of_storePara = train_result_dir + '/%s_Parameters.json'%(str(task_id)+'_'+str(job_id)+'_'+model_name)
    dir_of_storeModel = train_result_dir + '/%s_model'%(str(task_id)+'_'+str(job_id)+'_'+model_name)

    # 配置spark客户端
    sess = SparkSession\
        .builder\
        .master("local[4]")\
        .appName("randomforest_spark")\
        .config("spark.some.config.option", "some-value")\
        .getOrCreate()
    sc=sess.sparkContext
    sc.setLogLevel("ERROR")

    if options == 'train':
        time_start = time()
        #获取数据
        dataset = pd.read_csv(dir_of_inputdata)
        #用于测试 
        #dataset = dataset[0:1000]
        #限制多数类的数据
        #dataset = too.CalcMostLabel(dataset,Y_names)
        Y_datavec = dataset[Y_names].values
        #输出每个标签的数量
        print 'Counter:original y',Counter(Y_datavec)
        print'----------------------------------------------'
        #分别获得字符字段和数值型字段数据，且合并
        X_datavec,X_columns,vocabset,datavec_show_list= too.Merge_form(dataset,names_str,names_num,names_show,'vocabset','open')
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

        features = pd.DataFrame(X,) 
        targets = pd.DataFrame(Y, columns = ['Y'])
        #合拼矩阵
        merged = pd.concat([features, targets], axis = 1)
        #创建spark DataFrame
        raw_df = sess.createDataFrame(merged)
        #提取特征与目标
        fomula = RFormula(formula = 'Y ~ .', featuresCol="features",labelCol="label")
        raw_df = fomula.fit(raw_df).transform(raw_df)
        #拆分训练集和测试集
        xy_train, xy_test = raw_df.randomSplit([train_size, test_size],seed=666)
        #调用模型
        clf_model = dmp.Distr_RandomForestClassifier(xy_train,xy_test)
        #保存模型参数
        clf_model.write().overwrite().save(dir_of_storeModel)
        print'----------------------------------------------'
        dmp.Predict_test_data(xy_test, datavec_show_list, names_show, clf_model, dir_of_outputdata)
        duration = too.Duration(time()-time_start)
        print 'Total run time: %s'%duration

    if options == 'predict':
        time_start = time()
        with open(dir_of_storePara,'r') as f:
            para_dict = json.load(f)
        vocabset = para_dict['vocabset']
        ret_num = para_dict['ret_num']
        #获取数据
        dataset = pd.read_csv(dir_of_inputdata)
        #分别获得字符字段和数值型字段数据，且合并
        X_datavec,datavec_show_list = too.Merge_form(dataset,names_str,names_num,names_show,vocabset,'close')
        #数据归一化
        X = too.Data_process(X_datavec,normalized_type)
        #PCA降维
        if open_pca == 'open_pca':
            X = mlp.Model_PCA(X,ret_num)

        print'-------------Pdedict data shape---------------'
        print 'X.shape:',X.shape
        print'----------------------------------------------'
        print'--------------Start %s model------------------'%model_name

        features = pd.DataFrame(X,)
        #创建spark DataFrame
        raw_features = sess.createDataFrame(features)
        raw_x = VectorAssembler(inputCols=raw_features.columns,outputCol='features').transform(raw_features)
        clf_model = RandomForestClassificationModel.load(dir_of_storeModel)
        dmp.Predict_data(raw_x, datavec_show_list, names_show, clf_model, dir_of_outputdata)
        duration = too.Duration(time()-time_start)
        print 'Total run time: %s'%duration

if __name__ == "__main__":
    main()


