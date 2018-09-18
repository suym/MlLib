#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Parse_Data.py
用于解析数据的工具包
'''

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import re
import numpy as np
import pandas as pd


def Parse_cmd(dir_of_inputdata,options='one'):
    '''
    对输入的命令行数据进行解析
    dir_of_inputdata：输入的文本数据
    options：选择文本解析模式，'one'代表普通模式，'two'代表只选命令行中的关键词
    '''
    dataset = []
    remove_data = ['ls','cd','pwd','cat','ll']
    
    if options == 'one':
        #匹配非字母字符，即匹配特殊字符
        regEx = re.compile('\\W*')
        with open(dir_of_inputdata) as f:
            for line in f.readlines():
                #去掉行尾换行符
                line=line.rstrip('\n')
                listoftoken = regEx.split(line)
                #去掉空格值，且将字符串转变为小写
                tem = [tok.lower() for tok in listoftoken if len(tok)>0]
                #去掉第一个无用的序列号
                del tem[0]
                #去掉字符小于1的值
                tem2 = [b for b in tem if len(b)>1]
                #去掉无用的命令
                tem_data = [a for a in tem2 if a in remove_data]
                #len(data_tem4) != 0是为了去掉空集
                if len(tem_data) == 0 and len(tem2) != 0:
                    dataset.append(tem2)
    if options == 'two':
        regEx = re.compile('\\W*')
        with open(dir_of_inputdata) as f:
            for line in f.readlines():
                #去掉行尾换行符
                line=line.rstrip('\n')
                #按照空格划分字符串
                listoftoken = re.split(' ',line)
                data_tem = []
                for token in listoftoken:
                    #按照/划分一条命令行的字符串
                    tem = re.split(r'/',token)
                    #len(tok)>0是为了能取出/bin/read/ 中的read而不是最后一个/后的空格
                    tem2 = [tok for tok in tem if len(tok)>0]
                    if len(tem2) != 0:
                        #取出一个命令行中关键的命令字段
                        tem3 = tem2[-1]
                    else :
                        #如果token只有/或者空格，例如/ 或者//，那么tem、tem2为空
                        continue
                    #tem3是字符串，不是list，所以用append，而不是extend
                    data_tem.append(tem3)
                #将data_tem中的关键命令字段按照空格相连
                data_tem1 = ' '.join(data_tem)
                
                data_tem2 = regEx.split(data_tem1)
                #去掉空格的值，且将字符串转变为小写,注意不要写len(tok)>1,会对del data_tem[0]有影响，因为1到9序列号字符为1
                data_tem3 = [tok.lower() for tok in data_tem2 if len(tok)>0]
                #去掉第一个无用的序列号
                del data_tem3[0]
                #去掉字符小于1的值
                data_tem4 = [tok for tok in data_tem3 if len(tok)>1]
                #去掉无用的命令
                tem_data = [a for a in data_tem4 if a in remove_data]
                #len(data_tem4) != 0是为了去掉空集
                if len(tem_data) == 0 and len(data_tem4) != 0:
                    dataset.append(data_tem4)
        
    return dataset

def Parse_login(dir_of_inputdata):
    '''
    对输入的登录数据进行解析
    dir_of_inputdata：输入的文本数据
    '''
    dataset = []
    dataset_s = []
    num_start = 0
    num_end = 100
    data_tem = pd.read_csv(dir_of_inputdata,encoding="gbk")
    #取出时间列中的小时,分钟
    time_hour= pd.to_datetime(data_tem.iloc[:,5],format='%H:%M:%S').dt.hour
    #time_minute= pd.to_datetime(data_tem.iloc[:,5],format='%H:%M:%S').dt.minute
    #minute2hour=time_minute/60
    #data_tem[u'时间'] = time_hour+minute2hour.round(1)
    data_tem[u'时间'] = time_hour
    data_tem2 = data_tem.iloc[:,[1,2,3,4,5,6]]
    #原地填充空值
    data_tem2.fillna('my_NaN',inplace=True)
    #data_tem2[u'备用用户号'] = data_tem2[u'用户号']
    #按照用户号分组
    datagb = data_tem2.groupby(u'用户号')
    users_data = set(data_tem2[u'用户号'])
    print 'the number of users %s: '%len(users_data)
    data_tem3 = {}
    data_tem3_s = {}
    data_tem4 = []
    data_tem4_s = []
    for user in users_data:
        data_tem3[user] = datagb.get_group(user).drop([u'日期',u'操作'],axis=1).values
        data_tem3_s[user] = datagb.get_group(user).values
    for key_data,values_data in data_tem3.items():
        data_tem4.append(values_data.tolist())
        data_tem4_s.append(data_tem3_s[key_data].tolist())
            
    data_tem5 = data_tem4[num_start:num_end]
    for some_data in data_tem5:
        for some in some_data:
            dataset.append(some)
    #为了以后可以关联显示原始数据       
    data_tem5_s = data_tem4_s[num_start:num_end]
    for some_data_s in data_tem5_s:
        for some_s in some_data_s:
            dataset_s.append(some_s)
            
    return dataset,dataset_s

def Data_preprocess_login(dir_of_inputdata,dir_of_outputdata):
    '''
    对输入的登录数据进行词频预处理
    dir_of_inputdata：输入的文本数据
    dir_of_outputdata：输出的文本数据
    '''
    
    data_tem = pd.read_csv(dir_of_inputdata,encoding="gbk")
    #取出时间列中的小时,分钟
    time_hour= pd.to_datetime(data_tem.iloc[:,5],format='%H:%M:%S').dt.hour
    data_tem[u'时间'] = time_hour
    #按日期分组
    datagb = data_tem.groupby(u'日期')
    users_date = set(data_tem[u'日期'])
    data_tem2 = []
                     
    for date in users_date:
        tem = []
        data_tem = datagb.get_group(date)
        num_instit = len(data_tem[u'机构'].value_counts())
        num_user = len(data_tem[u'用户号'].value_counts())
        num_login = len(data_tem[u'登录设备号'].value_counts())
        num_diff1 = abs(num_instit-num_user)
        num_diff2 = abs(num_user-num_login)
        num_wrong_words = len(data_tem[u'报错码'].value_counts())
        num_sum_wrong_words = sum(data_tem[u'报错码'].value_counts())
        num_lgn = data_tem[u'操作'].value_counts()['USUSRLGN']
        num_ext = data_tem[u'操作'].value_counts()['USUSREXT']
        '''
        tem.append(num_instit)
        tem.append(num_user)
        tem.append(num_login)
        
        tem.append(num_wrong_words)
        tem.append(num_lgn)
        tem.append(num_ext)
        
        
        tem.append(num_diff1)'''
        #tem.append(num_diff2)
        tem.append(num_sum_wrong_words)
        
        time_index = list(data_tem[u'时间'].value_counts().index)
        time_range= range(1,25)
        names = locals()
        for time_t in time_range:
            if time_t in time_index:
                names['num_%s' % time_t] = data_tem[u'时间'].value_counts()[time_t]
            else:
                names['num_%s' % time_t] = 0
            #tem.append(eval('num_%s' % time_t))
        tem.append(date)
        data_tem2.append(tem)
    name1 = [u'机构',u'用户号',u'登录设备号',u'报错码',u'登入次数',u'登出次数']
    name3 = [u'报错码',u'登入次数',u'登出次数']
    name4 = [u'机构-用户号',u'用户号-登录设备号',u'报错码总计',u'日期']
    name5 = [u'报错码总计',u'日期']
    name2 =  [ u'1点',u'2点',u'3点',u'4点',u'5点',u'6点',u'7点',u'8点',u'9点',u'10点',u'11点',u'12点',
           u'13点',u'14点',u'15点',u'16点',u'17点',u'18点',u'19点',u'20点',u'21点',u'22点',
           u'23点',u'24点',u'日期']   
    names = name5
    dataset = pd.DataFrame(data_tem2,columns=names)
    dataset.to_csv(dir_of_outputdata,encoding='gbk')
    
def Data_preprocess_login_2(dir_of_inputdata,dir_of_outputdata):
    '''
    对输入的登录数据进行词频预处理
    dir_of_inputdata：输入的文本数据
    dir_of_outputdata：输出的文本数据
    '''
    
    data_tem = pd.read_csv(dir_of_inputdata,encoding="gbk")
    #取出时间列中的小时,分钟
    time_hour= pd.to_datetime(data_tem.iloc[:,5],format='%H:%M:%S').dt.hour
    data_tem[u'时间'] = time_hour
    
    users_tem = set(data_tem[u'用户号'])
    data_tem2 = []
    
    datagb_user = data_tem.groupby(u'用户号')
    
    for user in users_tem:
        #按用户号分组
        data_tem_user = datagb_user.get_group(user)
        datagb_date = data_tem_user.groupby(u'日期')
        date_tem = set(data_tem_user[u'日期'])
        for date in date_tem:
            tem = []
            #按日期分组
            data_tem_date = datagb_date.get_group(date)
            num_instit = len(data_tem_date[u'机构'].value_counts())
            num_of = sum(data_tem_date[u'机构'].value_counts())
            num_login = len(data_tem_date[u'登录设备号'].value_counts())
            num_wrong_words = len(data_tem_date[u'报错码'].value_counts())
            num_sum_wrong_words = sum(data_tem_date[u'报错码'].value_counts())
            if 'USUSRLGN' in list(data_tem_date[u'操作'].value_counts().index):
                num_lgn = data_tem_date[u'操作'].value_counts()['USUSRLGN']
            else:
                num_lgn = 0
            if 'USUSREXT' in list(data_tem_date[u'操作'].value_counts().index):
                num_ext = data_tem_date[u'操作'].value_counts()['USUSREXT']
            else:
                num_ext = 0
            
            tem.append(num_instit)
            tem.append(num_login)
        
            tem.append(num_wrong_words)
            tem.append(num_sum_wrong_words)
            tem.append(num_lgn)
            tem.append(num_ext)
        
        
            tem.append(num_of)
            
            time_index = list(data_tem_date[u'时间'].value_counts().index)
            time_range= range(1,25)
            names = locals()
            for time_t in time_range:
                if time_t in time_index:
                    names['num_%s' % time_t] = data_tem_date[u'时间'].value_counts()[time_t]
                else:
                    names['num_%s' % time_t] = 0
                tem.append(eval('num_%s' % time_t))
            tem.append(date)
            tem.append(user)
            data_tem2.append(tem)
            
    name1 = [u'机构种类数',u'登录设备号种类数',u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    name2 =  [ u'1点',u'2点',u'3点',u'4点',u'5点',u'6点',u'7点',u'8点',u'9点',u'10点',u'11点',u'12点',
           u'13点',u'14点',u'15点',u'16点',u'17点',u'18点',u'19点',u'20点',u'21点',u'22点',
           u'23点',u'24点']   
    name3 = [u'日期',u'用户号']
    names = name1+name2+name3
    dataset = pd.DataFrame(data_tem2,columns=names)
    dataset.to_csv(dir_of_outputdata,index=False,encoding='gbk')
        
def Data_preprocess_login_3(dir_of_inputdata,dir_of_outputdata):
    '''
    对输入的登录数据进行词频预处理
    dir_of_inputdata：输入的文本数据
    dir_of_outputdata：输出的文本数据
    '''
    
    data_tem = pd.read_csv(dir_of_inputdata,encoding="gbk")
    #取出时间列中的小时,分钟
    time_hour= pd.to_datetime(data_tem.iloc[:,5],format='%H:%M:%S').dt.hour
    data_tem[u'时间'] = time_hour
    
    users_tem = set(data_tem[u'登录设备号'])
    data_tem2 = []
    
    datagb_user = data_tem.groupby(u'登录设备号')
    
    for user in users_tem:
        #按登录设备号分组
        data_tem_user = datagb_user.get_group(user)
        datagb_date = data_tem_user.groupby(u'日期')
        date_tem = set(data_tem_user[u'日期'])
        for date in date_tem:
            tem = []
            #按日期分组
            data_tem_date = datagb_date.get_group(date)
            num_instit = len(data_tem_date[u'机构'].value_counts())
            num_of = sum(data_tem_date[u'机构'].value_counts())
            num_login = len(data_tem_date[u'用户号'].value_counts())
            num_wrong_words = len(data_tem_date[u'报错码'].value_counts())
            num_sum_wrong_words = sum(data_tem_date[u'报错码'].value_counts())
            if 'USUSRLGN' in list(data_tem_date[u'操作'].value_counts().index):
                num_lgn = data_tem_date[u'操作'].value_counts()['USUSRLGN']
            else:
                num_lgn = 0
            if 'USUSREXT' in list(data_tem_date[u'操作'].value_counts().index):
                num_ext = data_tem_date[u'操作'].value_counts()['USUSREXT']
            else:
                num_ext = 0
            
            tem.append(num_instit)
            tem.append(num_login)
        
            tem.append(num_wrong_words)
            tem.append(num_sum_wrong_words)
            tem.append(num_lgn)
            tem.append(num_ext)
        
        
            tem.append(num_of)
            
            time_index = list(data_tem_date[u'时间'].value_counts().index)
            time_range= range(1,25)
            names = locals()
            for time_t in time_range:
                if time_t in time_index:
                    names['num_%s' % time_t] = data_tem_date[u'时间'].value_counts()[time_t]
                else:
                    names['num_%s' % time_t] = 0
                tem.append(eval('num_%s' % time_t))
            tem.append(date)
            tem.append(user)
            data_tem2.append(tem)
            
    name1 = [u'机构种类数',u'用户号种类数',u'报错码种类数',u'报错码总数',u'登入次数',u'登出次数',u'登入登出总数目']
    name2 =  [ u'1点',u'2点',u'3点',u'4点',u'5点',u'6点',u'7点',u'8点',u'9点',u'10点',u'11点',u'12点',
           u'13点',u'14点',u'15点',u'16点',u'17点',u'18点',u'19点',u'20点',u'21点',u'22点',
           u'23点',u'24点']   
    name3 = [u'日期',u'登录设备号']
    names = name1+name2+name3
    dataset = pd.DataFrame(data_tem2,columns=names)
    dataset.to_csv(dir_of_outputdata,index=False,encoding='gbk')

    
