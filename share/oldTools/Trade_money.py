#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

def submain():
    dataset_1 = pd.read_csv('/besfs/users/suym/6.6.4.p01/Analysis/plot/python_doc/check_work/work/run/data/total_data.csv')
    names=['银行账户编号','交易套号','记账日期','交易记录时间','交易金额']
    dataset_1 = dataset_1[names]
    column_names = dataset_1.columns
    dataset_1['记账日期']=pd.to_datetime(dataset_1['记账日期'])
    dataset_1['交易记录时间']=pd.to_datetime(dataset_1['交易记录时间'])
    datagb_acc = dataset_1.groupby('银行账户编号')
    acc_nums = set(dataset_1['银行账户编号'])
    abnormal_data = []


    for acc_n in acc_nums:
        data_acc = datagb_acc.get_group(acc_n)
        datagb_time = data_acc.groupby('记账日期')
        acc_time =  set(data_acc['记账日期'])
        for acc_t in acc_time:
            data_time = datagb_time.get_group(acc_t)
            data_time = data_time.sort_values(by='交易记录时间')
            data_money = data_time['交易金额'].values.tolist()
            show_data = data_time.values.tolist()
            show_time = data_time['交易记录时间'].values.tolist()
            sum_money = 0
            key_open = 0

            while len(data_money)>0:
                trade_money = data_money.pop(0)
                show_data_tem = show_data.pop(0)
                show_time_tem = show_time.pop(0)
                sum_money = round(sum_money + trade_money,2)

                if (sum_money<0)&(key_open==0):
                    abnormal_data_tem = show_data_tem
                    abnormal_time = show_time_tem
                    key_open = key_open + 1
                if (sum_money>=0)&(key_open==1):
                    if (show_time_tem-abnormal_time)>15*10**9:
                        abnormal_data.append(abnormal_data_tem)
                    key_open = 0
                if (len(data_money)==0)&(key_open==1)&(sum_money<0):
                    abnormal_data.append(abnormal_data_tem)
         

    abnormal_data = pd.DataFrame(abnormal_data,columns=column_names)

    return abnormal_data

def main():
    abnormal_data = submain()
    abnormal_data.to_csv("./abnormal_data.csv",index=False)

if __name__ == "__main__":
    main() 

