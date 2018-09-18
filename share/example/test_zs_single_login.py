#!/usr/bin/python
# -*- coding: utf-8 -*-

import Tools as too
import ML_Package as mlp
import Parse_Data as pad
import pandas as pd

def hehe(tem):
    datavec_set_s = pd.read_csv('../pbs/zs_login/data/%s.csv'%tem,encoding='gbk')
    names1 = [u'日期',u'用户号']
    names2 = [ u'1点',u'2点',u'3点',u'4点',u'5点',u'6点',u'7点',u'8点',u'9点',u'10点',u'11点',u'12点',
               u'13点',u'14点',u'15点',u'16点',u'17点',u'18点',u'19点',u'20点',u'21点',u'22点',u'23点',u'24点']
    names3 = [u'机构种类数',u'登录设备号种类数',u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    names = names1+names2+[u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    names_show = names2+[u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    #names = names1+names2+[u'机构种类数',u'登入次数',u'登出次数',u'登入登出总数目']
    #names_show = names2+[u'机构种类数',u'登入次数',u'登出次数',u'登入登出总数目']
    #names = names1+names2+[u'机构种类数',u'登录设备号种类数',u'报错码种类数',u'报错码总数']
    #names_show = names2+[u'机构种类数',u'登录设备号种类数',u'报错码种类数',u'报错码总数']
    #names = names1+names2+[u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    #names_show = names2+[u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    #names = names1 +names3
    #names_show = names3
    datavec_set=datavec_set_s.drop(names, axis = 1)
    datavec_set_s=datavec_set_s.drop(names_show, axis = 1)
    datavec_set = datavec_set.values.tolist()
    datavec_set_s=datavec_set_s.values.tolist()
    vocabset=[1,2,3]

    #datavec_mod = too.Modify_counts_with_TFIDF(datavec_set)
    datavec,scaler = too.Data_process(datavec_set)

    best_epsilon,best_num = mlp.Gs_DBSCAN_parameter(datavec)
    clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,best_epsilon,best_num)
    #clst_labels,evaluate_score = mlp.Model_DBSCAN(datavec,0.001,2)
    
    if evaluate_score > 0.6:
        show_data = too.Find_exception_reason(datavec,datavec_set_s,clst_labels)
        too.Store_data('../pbs/zs_login/log_institutions/test_institutions%s.log'%tem,show_data)
        too.Find_exception('../pbs/zs_login/log_institutions/except_institutions%s.log'%tem,show_data)
        print datavec_set[0:2]
    
def main():
    for count in xrange(774):
        print count
        try:
            hehe(count)
        except: 
            pass
if __name__ == "__main__":
    main()
