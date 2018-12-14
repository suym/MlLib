#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import sys  
sys.path.append("..")
import pandas as pd
from src import Draw_Tools as dt

def main_model(dir_of_dict):
    #dir_of_dict = sys.argv[1]
    bag = dt.Read_info(dir_of_dict)
    options,task_id,job_id,dir_of_inputdata,\
    dir_of_outputdata,data_size,A_col,B_col = bag
    #dir_of_outputdata = dir_of_outputdata + '/%s.svg'%(str(task_id)+'_'+str(job_id)+'_'+options)

    dataset = pd.read_csv(dir_of_inputdata,encoding='utf-8')
    if data_size < len(dataset):
        dataset = dataset.sample(n=data_size,replace=False,axis=0)
    if options == 'AB_scatter':
        diagram = dt.AB_scatter(dataset,A_col,B_col)
        diagram.render_to_file(dir_of_outputdata)

if __name__ == '__main__':
    main_model(dir_of_dict)