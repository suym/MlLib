#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import sys  
sys.path.append("..")
import pandas as pd
from src import Feature_Tools as ft

def main():
    dir_of_dict = sys.argv[1]
    bag = ft.Read_info(dir_of_dict)
    options,task_id,job_id,dir_of_inputdata,\
    dir_of_outputdata,A_col,B_col,new_col = bag
    #dir_of_outputdata = dir_of_outputdata + '/%s.csv'%(str(task_id)+'_'+str(job_id)+'_'+options)

    dataset = pd.read_csv(dir_of_inputdata)
    if options == 'A_add_B':
        dataset = ft.A_add_B(dataset,A_col,B_col,new_col)
        dataset.to_csv(dir_of_outputdata,index=False)
    if options == 'A_minus_B':
        dataset = ft.A_minus_B(dataset,A_col,B_col,new_col)
        dataset.to_csv(dir_of_outputdata,index=False)
    if options == 'A_times_B':
        dataset = ft.A_times_B(dataset,A_col,B_col,new_col)
        dataset.to_csv(dir_of_outputdata,index=False)
    if options == 'A_divides_B':
        dataset = ft.A_divides_B(dataset,A_col,B_col,new_col)
        dataset.to_csv(dir_of_outputdata,index=False)

if __name__ == '__main__':
    main()