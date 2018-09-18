#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import DL_Package as dlp
import Parse_Data as pad
import pandas as pd

def main():
    #datavec_set_s = pd.read_csv('../run/data/userid.csv')
    datavec_set_s = pd.read_csv('../run/data/srcIp.csv')

    names = []
    names_show = []


    datavec_set=datavec_set_s.drop(names, axis = 1)
    datavec_set_s=datavec_set_s.drop(names_show, axis = 1)
    datavec_set = datavec_set[0:100000]
    datavec_set_s = datavec_set_s[0:100000]
    datavec_set = datavec_set.values.tolist()
    datavec_set_s=datavec_set_s.values.tolist()
    
    vocabset = too.CreateVocabList(datavec_set)
    datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    #datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
    
    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99%'])

    best_contamination = mlp.Gs_IsolationForest_parameter(datavec)
    clst_labels,scores_pred = mlp.Model_IsolationForest(datavec,best_contamination)
    #best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,2)
    #datavec = too.Data_inverse_transform(datavec,scaler)

    show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
    #too.Store_data('../run/log/sdtax_userid.log',show_data)
    #too.Find_exception('../run/log/except-sdtax_userid.log',show_data)
    #print datavec_set[0:2]
    
if __name__ == "__main__":
    main()
