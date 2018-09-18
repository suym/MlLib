#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import Parse_Data as pad
import pandas as pd

def hehe(tem):
    dir_file = 'account'
    log_file = 'Iforest_account_money'
    datavec_set_s = pd.read_csv('../pbs/zs_account/data_%s/%s.csv'%(dir_file,tem))
    #name_char = ['银行账户编号','套录代码','冲正类型代码','借贷方向代码','核算种类编号','资金清算代码','交易状态代码']
    name_char = ['交易金额','联机余额']
    name_total = name_char + ['记账日期','事件编号','银行限额账户编号']

    datavec_set=datavec_set_s[name_char]
    
    datavec_set = datavec_set.values.tolist()
    
    #vocabset = too.CreateVocabList(datavec_set)
    #datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    #datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
    
    datavec_set_s=datavec_set_s[name_total]
    datavec_set_s=datavec_set_s.values.tolist()

    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99%'])

    #best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,2)
    evaluate_score,best_contamination = mlp.Gs_IsolationForest_parameter(datavec)
    clst_labels,scores_pred = mlp.Model_IsolationForest(datavec,best_contamination)
    
    if evaluate_score >= 0.72:
        #show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
        show_data = too.Add_exception_scores(datavec_set_s,clst_labels,scores_pred)
        too.Store_data('../pbs/zs_account/log_%s/%s_%s.log'%(log_file,log_file,tem),show_data)
        print datavec_set[0:2]
    
def main():
    for count in xrange(128):
        print count
        try:
            hehe(count)
    except NameError: 
            pass
        
if __name__ == "__main__":
    main()
