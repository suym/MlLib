#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import json
import math

def Read_info(dir_of_dict):
    '''
    读取配置文件
    '''
    with open(dir_of_dict,'r') as f:
        column_lines = f.read()
        name_dict = eval(column_lines)
    A_func_B = ['A_add_B','A_minus_B','A_times_B','A_divides_B']
    A_func = ['A_log','A_squared']
    Muti_col = ['Cal_corr','Select_col']

    options = name_dict['options']
    task_id = name_dict['task_id']
    job_id = name_dict['job_id']
    dir_of_inputdata = name_dict['dir_of_inputdata']
    dir_of_outputdata = name_dict['dir_of_outputdata']

    if options in A_func_B:
        A_col = name_dict['A_col']
        B_col = name_dict['B_col']
        new_col = name_dict['new_col']
        bag = options,task_id,job_id,dir_of_inputdata,dir_of_outputdata,A_col,B_col,new_col
        
    if options in A_func:
        A_col = name_dict['A_col']
        new_col = name_dict['new_col']
        bag = options,task_id,job_id,dir_of_inputdata,dir_of_outputdata,A_col,new_col

    if options in Muti_col:
        corr_col = name_dict['corr_col']
        bag = options,task_id,job_id,dir_of_inputdata,dir_of_outputdata,corr_col

    return bag

def A_add_B(data,A,B,C):
    data[C] = data[A] + data[B]
    return data

def A_minus_B(data,A,B,C):
    data[C] = data[A] - data[B]
    return data

def A_times_B(data,A,B,C):
    data[C] = data[A] * data[B]
    return data

def A_divides_B(data,A,B,C):
    data[C] = data[A] / data[B]
    return data

def A_log(data,A,C):
    data[C] = data[A].map(lambda x: math.log(x))
    return data

def A_squared(data,A,C):
    data[C] = data[A].map(lambda x: x**2)
    return data

def A_corr_B(data,corr_col):
    data = data[corr_col]
    
    return data.corr()

def Select_col(data,corr_col):
    data = data[corr_col]

    return data

