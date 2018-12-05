#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Tools_Package.py
有用的工具包
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import operator
import copy
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------
# 数据预处理
# ---------------------------------------------

def Read_info(dir_of_dict,choose_type):
    '''
    读取配置文件
    '''
    with open(dir_of_dict,'r') as f:
        column_lines = f.read()
        name_dict = eval(column_lines)
    if choose_type == 'supervision':
        options = name_dict['names_type']
        task_id = name_dict['task_id']
        job_id = name_dict['job_id']
        train_result_dir = name_dict['train_result_dir']
        names_str = name_dict['names_str']
        names_num = name_dict['names_num']
        names_show = name_dict['names_show']
        Y_names = name_dict['Y_name']
        dir_of_inputdata = name_dict['dir_of_inputdata']
        dir_of_outputdata = name_dict['dir_of_outputdata']
        open_pca = name_dict['open_pca']
        train_size = name_dict['train_size']
        test_size = name_dict['test_size']
        normalized_type = name_dict['normalized_type']

        bag = name_dict,options,task_id,job_id,train_result_dir,\
              names_str,names_num,names_show,Y_names,dir_of_inputdata,\
              dir_of_outputdata,open_pca,train_size,test_size,normalized_type
    if choose_type == 'non-supervision':
        task_id = name_dict['task_id']
        job_id = name_dict['job_id']
        train_result_dir = name_dict['train_result_dir']
        names_str = name_dict['names_str']
        names_num = name_dict['names_num']
        names_show = name_dict['names_show']
        dir_of_inputdata = name_dict['dir_of_inputdata']
        dir_of_outputdata = name_dict['dir_of_outputdata']
        open_pca = name_dict['open_pca']
        normalized_type = name_dict['normalized_type']

        bag = name_dict,task_id,job_id,train_result_dir,\
              names_str,names_num,names_show,\
              dir_of_inputdata,dir_of_outputdata,open_pca,normalized_type

    return bag

def Data_process(x_ori,options='minmaxscaler'):
    '''
    对数据进行z-score标准化或者标准化到0到1
    x_ori：输入数据，要求类型为多维数组，行是样本数，列是特征值
    '''
    if options == 'z-score':
        scaler = StandardScaler()
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    if options == 'minmaxscaler':
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(x_ori)
        X = scaler.transform(x_ori)
    
    return X

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
    print'----------------------------------------------'

    return list(vocabset)

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
                print "the word : %s is not in my vocabulary!" % word
                #pass
        if sum(tem) > 0:
            datavec.append(tem)

    return datavec

def CalcMostLabel(xytable,Y_names,low_value = 0,hight_value = 30000):
    #按照类别的数量，从多到少排序
    Y_tem = sorted(Counter(xytable[Y_names]).items(),key=operator.itemgetter(1), reverse=True) 
    Y_tem2 = []
    Y_tem3 = []
    new_xy = pd.DataFrame()
    #只选择样本数大于low_value的类别数据
    for i in Y_tem:
        if (i[1] > low_value)&(i[1] < hight_value):
            Y_tem2.append(i[0])
        if i[1] > hight_value:
            Y_tem3.append(i[0])
    xy_tem = xytable[xytable[Y_names].isin(Y_tem2)]
    new_xy = pd.concat([new_xy,xy_tem])
    #限制多数类的数据量
    if len(Y_tem3) > 0: 
        for label in Y_tem3:
            Y_tem4 = xytable[xytable[Y_names]==label].sample(n=hight_value,random_state=0)
            new_xy = pd.concat([new_xy,Y_tem4])

    return new_xy

def Merge_form(dataset,names_str,names_num,names_show,vocab_set,key):
    '''
    将数值型和字符型的dataframe合并
    '''
    #分别获得字符字段和数值型字段数据
    dataset_str = dataset[names_str]
    dataset_num = dataset[names_num]
    dataset_show = dataset[names_show]
    dataset_str_list = dataset_str.values.tolist()
    datavec_num_list = dataset_num.values.tolist()
    datavec_show_list = dataset_show.values.tolist()
    if key == 'open':
        vocabset = CreateVocabList(dataset_str_list)
    else:
        vocabset = vocab_set
    datavec_str_list = BagofWords2Vec(vocabset,dataset_str_list)
    #vocabset_index = {y:i for i,y in enumerate(vocabset)}
    #将list转化为DataFrame，合并两表
    datavec_str = pd.DataFrame(datavec_str_list,columns=vocabset)
    datavec_num = pd.DataFrame(datavec_num_list,columns=names_num)
    #按照左表连接，右表可以为空
    data_tem = pd.merge(datavec_num,datavec_str,how="left",right_index=True,left_index=True)
    X_datavec = data_tem.values
    X_columns = data_tem.columns
    if key == 'open':
        return X_datavec,X_columns,vocabset,datavec_show_list
    else:
        return X_datavec,datavec_show_list

def StorePara(dir_of_storePara,vocabset,ret_num):
    para_dict = {}
    para_dict['vocabset']=vocabset
    para_dict['ret_num']=ret_num
    with open(dir_of_storePara,'w') as f:
        json.dump(para_dict,f)

def Predict_test_data(X_ori, Y, datavec_show_list, names_show,clf_model, dir_of_outputdata,options='ML',mtype='class'):
    if options=='ML':
        y_pred = clf_model.predict(X_ori)
    if options=='MFNN':
        y_pred = clf_model.predict(X_ori,batch_size=128)
    if mtype=='class':
        print'----------------------------------------------'
        if len(set(Y)) == 2:
            print "Accuracy score: ",accuracy_score(Y,y_pred,normalize=True)
            print "Precision score: ",precision_score(Y,y_pred)
            print "Recall score: ",recall_score(Y,y_pred)
        else:
            print "Accuracy score: ",accuracy_score(Y,y_pred,normalize=True)
            print "Precision score: ",precision_score(Y,y_pred,average='macro')
            print "Recall score: ",recall_score(Y,y_pred,average='macro')
        print'----------------------------------------------'
        y_com = [y_pred[i]==Y[i] for i in range(len(Y))]
        #左表可以为空，按照右表连接
        xy_tem = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                          pd.DataFrame(Y,columns=['y_true']),
                          how="right",right_index=True,left_index=True)
        xy_tem_1 = pd.merge(xy_tem,
                            pd.DataFrame(y_pred,columns=['y_pred']),
                            how="left",right_index=True,left_index=True)
        xy_table = pd.merge(xy_tem_1,
                            pd.DataFrame(y_com,columns=['y_true==y_pred']),
                            how="left",right_index=True,left_index=True)
        xy_table.to_csv(dir_of_outputdata,index=False)
    if mtype=='reg':
        y_com = [round(abs(y_pred[i]-Y[i]),2) for i in range(len(Y))]
        #左表可以为空，按照右表连接
        xy_tem = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                          pd.DataFrame(Y,columns=['y_true']),
                          how="right",right_index=True,left_index=True)
        xy_tem_1 = pd.merge(xy_tem,
                            pd.DataFrame(y_pred,columns=['y_pred']),
                            how="left",right_index=True,left_index=True)
        xy_table = pd.merge(xy_tem_1,
                            pd.DataFrame(y_com,columns=['abs(y_true-y_pred)']),
                            how="left",right_index=True,left_index=True)
        xy_table.to_csv(dir_of_outputdata,index=False)

def Predict_data(X, datavec_show_list, names_show, clf_model, dir_of_outputdata,options='ML',mtype='class'):
    if options=='ML':
        y_pred = clf_model.predict(X)
    if options=='MFNN':
        y_pred = clf_model.predict(X,batch_size=128)
    if mtype=='class':
        #y_pred[y_pred>90]=0
        #左表可以为空，按照右表连接
        xy_table = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                            pd.DataFrame(y_pred,columns=['y_pred']),
                            how="right",right_index=True,left_index=True)
        xy_table.to_csv(dir_of_outputdata,index=False)
    if mtype=='reg':
        #左表可以为空，按照右表连接
        xy_table = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                            pd.DataFrame(y_pred,columns=['y_pred']),
                            how="right",right_index=True,left_index=True)
        xy_table.to_csv(dir_of_outputdata,index=False)

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

def Find_exception_reason(dataset,dataset_s,clst_labels,column_names):
    '''
    找出异常用户的原因，并给出异常值
    dataset：输入数据，要求类型为多维数组，行是样本数，列是特征值
    dataset_s：输入数据，要求类型为pandas.DataFrame，行是样本数，列是特征值，比dataset多了用户和日期列，用于显示异常数据
    clst_labels：数据的聚类标签
    column_names:dataset的列名
    '''
    num_dataset = len(dataset)
    #深度复制，避免修改show_data会影响到dataset_s
    show_data = copy.deepcopy(dataset_s)
    
    #为了方便前端展示基准值
    for_base_value = copy.deepcopy(dataset_s[column_names])
    for_base_value.loc[:,'labels'] = clst_labels
    for_show_base_value = for_base_value[for_base_value["labels"]!=-1].agg(["mean"]).round(4)
    for_show_base_value = for_show_base_value.drop(['labels'],axis =1)
    
    data_tem = pd.DataFrame(dataset)
    finalTable=pd.merge(data_tem,pd.DataFrame(clst_labels,columns=["labels"]),how="left",left_index=True,right_index=True)
    
    #正常人的平均行为
    nomalTable=finalTable[finalTable["labels"]!=-1].agg(["mean"]).round(4)
    #去掉label的列
    nomalTable = nomalTable.drop(['labels'],axis =1)
        
    for data_index in range(num_dataset):
        #nomalTable.values[0]的[0]是为了去掉[[]]的一层括号,把每一个维度和正常的均值点偏差求和，并保留两位小数
        user_score =  round(abs((dataset[data_index]-nomalTable.values[0])).sum(),2)
        show_data.loc[data_index,'abnormality'] = user_score
        show_data.loc[data_index,'exceptionCategory'] = 'class'+str(clst_labels[data_index])
        #只获取异常数据
        exceptionTable = show_data[show_data['exceptionCategory']=='class-1']

    return exceptionTable,for_show_base_value

def Add_exception_reason(dataset_s,clst_labels,column_names,scores_pred):
    '''
    保存模型中已经计算出来的异常度
    dataset_s：输入数据，要求类型为pandas.DataFrame，行是样本数，列是特征值，比dataset多了用户和日期列，用于显示异常数据
    clst_labels：数据的聚类标签
    column_names：列名
    scores_pred:异常得分
    '''
    #深度复制，避免修改exceptionTable会影响到dataset_s
    exceptionTable = copy.deepcopy(dataset_s)
    exceptionTable.loc[:,'abnormality'] = scores_pred
    exceptionTable.loc[:,'exceptionCategory'] = clst_labels
    #为了方便前端展示基准值
    for_base_value = copy.deepcopy(dataset_s[column_names])
    for_base_value.loc[:,'labels'] = clst_labels
    for_show_base_value = for_base_value[for_base_value["labels"]!=-1].agg(["mean"]).round(4)
    for_show_base_value = for_show_base_value.drop(['labels'],axis =1)

    return exceptionTable,for_show_base_value

