#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

from mllib.classModel import Distr_GBTClassifier,Distr_LogisticRegression,Distr_RandomForestCla,\
MD_GradientBoostingCla,MD_LinearSVC,MD_LogisticRegression,MD_RandomForestCla,MD_SVC_linear,\
MD_SVC_rbf,MD_VotingClassifier,RE_GradientBoostingCla,RE_LogisticReg,RE_RandomForestCla
from mllib.clusterModel import MD_DBSCAN,MD_DeepAutoEncoder,MD_IsolationForest,MD_LocalOutlierFactor
from mllib.regressionModel import Distr_GBTRegressor,Distr_LinearRegression,Distr_RandomForestReg,\
MD_GradientBoostingReg_huber,MD_GradientBoostingReg_lslad,MD_Lasso,MD_LinearSVR,MD_RandomForestReg,\
MD_Ridge,MD_SVR_linear,MD_SVR_rbf,RE_GradientBoostingReg,RE_Lasso,RE_RandomForestReg,RE_Ridge
from mllib.feature import A_func,A_func_B,Cal_correlation,Select_column
from mllib.visualData import Pie_diagram,Scatter_diagram

def main():
    options = sys.argv[1]
    dir_of_dict = sys.argv[2]
    eval('%s.main_model'%options)(dir_of_dict)

if __name__ == '__main__':
    main()
