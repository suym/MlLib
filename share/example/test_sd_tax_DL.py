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
    datavec_set = datavec_set[0:20000]
    datavec_set_s = datavec_set_s[0:20000]
    datavec_set = datavec_set.values.tolist()
    datavec_set_s=datavec_set_s.values.tolist()
    
    vocabset = too.CreateVocabList(datavec_set)
    #datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
    
    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99%'])
    
    loss,X_diff_loss = dlp.Model_deep_auto_encoder(datavec)
    clst_labels = dlp.Gs_deep_auto_encoder_parameter(loss,X_diff_loss)
    
    show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
    too.Store_data('../run/log/sdtax_userid.log',show_data)
    too.Find_exception('../run/log/except-sdtax_userid.log',show_data)
    #print datavec_set[0:2]
    
if __name__ == "__main__":
    main()
