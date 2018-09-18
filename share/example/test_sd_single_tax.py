#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import Parse_Data as pad
import pandas as pd

def hehe(tem):
    dir_file = 'srcip_dstip_httpurl'
    #datavec_set_s = pd.read_csv('../pbs/sd_tax_2/data_%s/%s.csv'%(dir_file,tem))
    datavec_set_s = pd.read_csv('../pbs/sd_tax_2/data_%s/%s.csv'%('srcIp',tem))
    names = ['gwxh','gwssswjg','zndm','gndm','startatms']
    names_show = ['gwxh','gwssswjg','zndm','gndm']


    datavec_set=datavec_set_s.drop(names, axis = 1)
    datavec_set_s=datavec_set_s.drop(names_show, axis = 1)
    datavec_set = datavec_set.values.tolist()
    datavec_set_s=datavec_set_s.values.tolist()
    
    vocabset = too.CreateVocabList(datavec_set)
    #datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
    
    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99%'])

    best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,1)
    
    if evaluate_score > 0.8:
        show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
        too.Store_data('../pbs/sd_tax_2/log_%s/%s_%s.log'%(dir_file,dir_file,tem),show_data)
        #too.Find_exception('../pbs/sd_tax_2/log_%s/except_%s_%s.log'%(dir_file,dir_file,tem),show_data)
        #print datavec_set[0:2]
    
def main():
    #for count in xrange(371):
    for count in xrange(319,371):
        print count
        try:
            hehe(count)
        #except NameError: 
        except: 
            pass
if __name__ == "__main__":
    main()
