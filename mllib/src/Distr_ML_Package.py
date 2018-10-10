#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Distr_ML_Package.py
分布式机器学习包
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import pandas as pd
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,GBTClassifier

# ---------------------------------------------
# 分类算法
# ---------------------------------------------

def Distr_LogisticRegression(xy_train,xy_test):
    lr = LogisticRegression()
    evalu = BinaryClassificationEvaluator()
    grid_1 = ParamGridBuilder()\
            .addGrid(lr.regParam, [1.0])\
            .addGrid(lr.elasticNetParam, [0.0,0.3,0.5,0.8,1.0])\
            .build()    
    cv_1 = CrossValidator(estimator=lr,estimatorParamMaps=grid_1,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel_1=cv_1.fit(xy_train)
    print "Grid scores: "
    best_params_1 = Get_best_params(cvModel_1)['elasticNetParam']
    grid = ParamGridBuilder()\
            .addGrid(lr.regParam, [0.001,0.01,0.2,1.0,8.0,50.0])\
            .addGrid(lr.elasticNetParam, [best_params_1,])\
            .build() 
    cv = CrossValidator(estimator=lr,estimatorParamMaps=grid,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel=cv.fit(xy_train)
    best_params = Get_best_params(cvModel)

    print "Best parameters set found: %s" % best_params
    
    return cvModel.bestModel

def Distr_GBTClassifier(xy_train,xy_test):
    gf = GBTClassifier(minInstancesPerNode=20,maxDepth=25)
    evalu = BinaryClassificationEvaluator()
    grid_1 = ParamGridBuilder()\
            .addGrid(gf.maxIter, [100])\
            .addGrid(gf.subsamplingRate, [0.5,0.8,1.0])\
            .build()    
    cv_1 = CrossValidator(estimator=gf,estimatorParamMaps=grid_1,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel_1=cv_1.fit(xy_train)
    print "Grid scores: "
    best_params_1 = Get_best_params(cvModel_1)['subsamplingRate']
    grid = ParamGridBuilder()\
            .addGrid(gf.maxIter, [300,500])\
            .addGrid(gf.subsamplingRate, [best_params_1,])\
            .build() 
    cv = CrossValidator(estimator=gf,estimatorParamMaps=grid,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel=cv.fit(xy_train)
    best_params = Get_best_params(cvModel)

    print "Best parameters set found: %s" % best_params
    
    return cvModel.bestModel

def Distr_RandomForestClassifier(xy_train,xy_test):
    rf = RandomForestClassifier(minInstancesPerNode=20,maxDepth=25)
    evalu = BinaryClassificationEvaluator()
    grid_1 = ParamGridBuilder()\
            .addGrid(rf.numTrees, [100])\
            .addGrid(rf.featureSubsetStrategy, ['0.5','0.8','1.0'])\
            .build()    
    cv_1 = CrossValidator(estimator=rf,estimatorParamMaps=grid_1,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel_1=cv_1.fit(xy_train)
    print "Grid scores: "
    best_params_1 = Get_best_params(cvModel_1)['featureSubsetStrategy']
    grid = ParamGridBuilder()\
            .addGrid(rf.numTrees, [300,500])\
            .addGrid(rf.featureSubsetStrategy, [best_params_1,])\
            .build() 
    cv = CrossValidator(estimator=rf,estimatorParamMaps=grid,evaluator=evalu,numFolds=5)
    #寻找模型的最佳组合参数,cvModel将返回估计的最佳模型
    cvModel=cv.fit(xy_train)
    best_params = Get_best_params(cvModel)

    print "Best parameters set found: %s" % best_params
    
    return cvModel.bestModel

# ---------------------------------------------
# 函数
# ---------------------------------------------

def Print_class_info(xy_predict):
    '''
    打印和分类效果有关的信息
    xy_predict:模型预测的数据集
    '''
    def build_predict_target(row):
        return (float(row.prediction), float(row.label))
    predict_and_target_rdd = xy_predict.rdd.map(build_predict_target)
    metrics = BinaryClassificationMetrics(predict_and_target_rdd)

    correct_amount = xy_predict.filter(xy_predict['label'] == xy_predict['prediction']).count() 
    total_amount = xy_predict.count()
    accuracy_rate = float(correct_amount)/total_amount
    positive_precision_amount = xy_predict.filter(xy_predict['label'] == 1).filter(xy_predict['prediction'] == 1).count()
    positive_amount = xy_predict.filter(xy_predict['label'] == 1).count()
    predict_amount = xy_predict.filter(xy_predict['prediction'] == 1).count()

    recall_rate =  float(positive_precision_amount)/positive_amount
    precision_rate = float(positive_precision_amount)/predict_amount

    print'----------------------------------------------'
    print "Precision score: %s" % precision_rate
    print "Recall score: %s" % recall_rate
    print "Accuracy score: %s" % accuracy_rate
    print "Area under PR: %s" % metrics.areaUnderPR
    print "Area under ROC: %s" % metrics.areaUnderROC
    print'----------------------------------------------'

def Get_best_params(cvModel):
    '''
    获得交叉验证最优的参数
    cvModel:交叉验证类对象
    '''
    results = [
        (
            [
                {key.name: paramValue} 
                for key, paramValue 
                in zip(
                    params.keys(), 
                    params.values())
            ], metric
        ) 
        for params, metric 
        in zip(
            cvModel.getEstimatorParamMaps(), 
            cvModel.avgMetrics
        )
             ]
    best_params_tem = sorted(results, 
                   key=lambda el: el[1], 
                   reverse=True)[0][0]
    best_params = []
    for k in best_params_tem:
        best_params.extend(k.items())

    for kk in results:
        print "\t%0.3f for %s"%(kk[1],kk[0])
    print'----------------------------------------------'

    return dict(best_params)

def Predict_test_data(xy_test, datavec_show_list, names_show, clf_model, dir_of_outputdata):
    xy_predict = clf_model.transform(xy_test)
    Print_class_info(xy_predict)
    xy_select = xy_predict.select("label","probability", "prediction").toPandas()
    #左表可以为空，按照右表连接
    xy_table = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                          xy_select,
                          how="right",right_index=True,left_index=True)
    xy_table.to_csv(dir_of_outputdata,index=False)

def Predict_data(X, datavec_show_list, names_show, clf_model, dir_of_outputdata):
    xy_predict = clf_model.transform(X)
    xy_select = xy_predict.select("probability", "prediction").toPandas()
    #左表可以为空，按照右表连接
    xy_table = pd.merge(pd.DataFrame(datavec_show_list,columns=names_show),
                          xy_select,
                          how="right",right_index=True,left_index=True)
    xy_table.to_csv(dir_of_outputdata,index=False)

    

