#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pygal              
from collections import Counter

def Read_info(dir_of_dict):
    '''
    读取配置文件
    '''
    with open(dir_of_dict,'r') as f:
        column_lines = f.read().encode('utf-8')
        name_dict = eval(column_lines)
    A_func_B = ['AB_scatter',]
    A_func = ['A_pie',]

    options = name_dict['options']
    task_id = name_dict['task_id']
    job_id = name_dict['job_id']
    dir_of_inputdata = name_dict['dir_of_inputdata']
    dir_of_outputdata = name_dict['dir_of_outputdata']
    data_size = name_dict['data_size']

    if options in A_func_B:
        A_col = name_dict['A_col'].decode('utf-8')
        B_col = name_dict['B_col'].decode('utf-8')
        bag = options,task_id,job_id,dir_of_inputdata,dir_of_outputdata,data_size,A_col,B_col
        
    if options in A_func:
        A_col = name_dict['A_col'].decode('utf-8')
        bag = options,task_id,job_id,dir_of_inputdata,dir_of_outputdata,data_size,A_col

    return bag

def AB_scatter(data,A,B):
    xy = zip(data[A],data[B])
    xy_chart = pygal.XY(stroke=False)
    xy_chart.title = 'Correlation'
    xy_chart.x_title = A
    xy_chart.y_title = B
    xy_chart.add('%s-%s'%(A,B), xy)
    return xy_chart

def A_pie(data,A):
    x = dict(Counter(data[A]))
    pie_chart = pygal.Pie()
    pie_chart.title = '%s Pie'%A  
    for key,value in x.items():
        pie_chart.add(str(key), value)
    return pie_chart 