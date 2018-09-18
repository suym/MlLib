#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import Parse_Data as pad
import pandas as pd

def main():
    datavec_set_s = pd.read_csv('../pbs/zs_account/data_account/10.csv')
    
    name_char = ['银行限额账户编号','银行账户编号','交易套内序号','套录代码','交易序号','借贷方向代码','核算种类编号','资金清算代码',
                 '易摘要代码','2G交易摘要代码','产品编号','交易状态代码']
    #name_num = ['交易金额','联机余额']
    name_num = []
    name_total = name_char + name_num + ['记账日期','事件编号']

    names = ['事件编号','银行限额账户修饰符','银行账户修饰符','币种代码','分行机构编号','核算机构编号','总账机构编号',
            '交易类型代码','交易流水号','记账日期','交易记录时间','起息日期','积数代码','积数金额','银行账户交易查询级别代码',
            '记账类型代码','扩展币种代码','扩展金额','核对币种代码','核对金额','辅助币种代码','辅助金额','摘要币种代码','摘要金额',
            '统计币种代码','统计金额','统计交易摘要代码','分析币种代码','分析金额','分析交易摘要代码','附加币种代码','附加金额',
             '附加交易摘要代码','业务日期','插入日期','插入时间','更新日期','更新时间','来源系统','来源表']
    names_show = ['事件编号','银行限额账户修饰符','银行账户修饰符','币种代码','分行机构编号','核算机构编号','总账机构编号',
            '交易类型代码','交易流水号','记账日期','交易记录时间','起息日期','积数代码','积数金额','银行账户交易查询级别代码',
            '记账类型代码','扩展币种代码','扩展金额','核对币种代码','核对金额','辅助币种代码','辅助金额','摘要币种代码','摘要金额',
            '统计币种代码','统计金额','统计交易摘要代码','分析币种代码','分析金额','分析交易摘要代码','附加币种代码','附加金额',
             '附加交易摘要代码','业务日期','插入日期','插入时间','更新日期','更新时间','来源系统','来源表']


    #datavec_set=datavec_set_s.drop(names, axis = 1)
    #datavec_set_s=datavec_set_s.drop(names_show, axis = 1)
    
    datavec_set=datavec_set_s[name_char]
    
    datavec_set = datavec_set[0:200000]
    datavec_set_s = datavec_set_s[0:200000]
    datavec_set = datavec_set.values.tolist()
    
    vocabset = too.CreateVocabList(datavec_set)
    datavec_set = too.PolyofWords2Vec(vocabset,datavec_set)
    #datavec_set = too.SetofWords2Vec(vocabset,datavec_set)
   
    data_tem = pd.DataFrame(datavec_set)
    data_tem3 = pd.DataFrame(datavec_set_s[name_num].values.tolist())
    data_tem2= pd.merge(data_tem,data_tem3,how="left",right_index=True,left_index=True)
    datavec_set = data_tem2.values.tolist()
    
    datavec_set_s=datavec_set_s[name_total]
    datavec_set_s=datavec_set_s.values.tolist()

    datavec,scaler = too.Data_process(datavec_set)
    #pca_num,ret = mlp.Gs_PCA(datavec)
    #print pca_num,ret
    #datavec = mlp.Model_PCA(datavec,ret['99.9%'])

    #best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,2)
    
    best_contamination = mlp.Gs_IsolationForest_parameter(datavec)
    clst_labels,scores_pred = mlp.Model_IsolationForest(datavec,best_contamination)

    #datavec = too.Data_inverse_transform(datavec,scaler)
    show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
    too.Store_data('../run/log/zs_account.log',show_data)
    print datavec_set[0:2]
    
if __name__ == "__main__":
    main()
