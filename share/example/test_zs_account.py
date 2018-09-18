#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import Parse_Data as pad
from time import time
import pandas as pd

def main():
    time_start = time()
    datavec_set_s = pd.read_csv('../run/data/account_with_if_nxh_total.csv')
    datavec_set_s = datavec_set_s[datavec_set_s['ct']>1]
    
    names = ['tradeNum','neixuhao','time','sum1','sum2']
    names_show =['neixuhao','sum1','sum2']
    #names = ['sum1']
    #names_show =['sum1','tradeNum','time']


    #datavec_set=datavec_set_s[names]
    #datavec_set_s=datavec_set_s[names_show]
    
    datavec_set=datavec_set_s.drop(names, axis = 1)
    datavec_set_s=datavec_set_s.drop(names_show, axis = 1)
    zs_columns=list(datavec_set.columns)
    
    
    #datavec_set = datavec_set[0:200000]
    #datavec_set_s = datavec_set_s[0:200000]
    datavec_set = datavec_set.values.tolist()
    datavec_set_s=datavec_set_s.values.tolist()
    
    #vocabset = too.CreateVocabList(datavec_set)
    #datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    #datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
   
    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99.9%'])

    best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,1)
    
    #evaluate_score,best_contamination = mlp.Gs_IsolationForest_parameter(datavec)
    #clst_labels,scores_pred = mlp.Model_IsolationForest(datavec,best_contamination)

    #best_neighbor,best_contamination = mlp.Gs_LocalOutlierFactor_parameter(datavec)
    #clst_labels,scores_pred = mlp.Model_LocalOutlierFactor(datavec,best_neighbor,best_contamination)

    #datavec = too.Data_inverse_transform(datavec,scaler)
    #show_data = too.Show_data(zs_columns,datavec,clst_labels)
    show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
    #too.Store_data('../run/log/zs_24_account_sum2_DBSCAN.log',show_data)
    #too.Store_data('../run/log/zs_total_columns_DBSCAN.log',show_data)
    too.Store_data('../run/log/zs_DBSCAN.log',show_data)
    #show_data = too.Add_exception_scores(datavec_set_s,clst_labels,scores_pred)
    #too.Store_data('../run/log/zs_24_account_sum2_iforest.log',show_data)
    #too.Store_data('../run/log/zs_24_account_sum2_lof.log',show_data)
    print datavec_set[0:2]
    #dataviz = mlp.Model_TSNE(datavec,2)
    #too.Visualize(dataviz,clst_labels,2)
    duration = too.Duration(time()-time_start)
    print duration
    
if __name__ == "__main__":
    main()
