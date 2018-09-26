#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
ML_Package_for_HarBin.py
机器学习的工具包
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression,LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC,LinearSVR, SVR

from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.decomposition import PCA, IncrementalPCA

from sklearn.cluster import DBSCAN,KMeans
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE


# ---------------------------------------------
# 数据不平衡问题
# ---------------------------------------------

def KMeans_unbalanced(X_datavec,Y_datavec,X_columns,Y_names,num_used=20000):

    XY_datavec = pd.merge(pd.DataFrame(X_datavec,columns=X_columns),
                          pd.DataFrame(Y_datavec,columns=[Y_names]),
                          how="left",right_index=True,left_index=True)
    XY_datavec_normal = XY_datavec[XY_datavec[Y_names]==0]
    X_datavec_normal = XY_datavec_normal.drop(Y_names, axis = 1).values.tolist()
    XY_datavec_outlier = XY_datavec[XY_datavec[Y_names]==1]
    
    #处理数据不平衡问题
    best_num_cluster = GS_KMeans_parameter(X_datavec_normal)
    y_clst_labels = Model_KMeans(X_datavec_normal,best_num_cluster)
    #避免和已经标记label重合
    y_clst_labels = [i+100 for i in y_clst_labels]
    print 'y_clst_labels Information:',set(y_clst_labels)
    print'----------------------------------------------'
    XY_clst_normal = pd.merge(pd.DataFrame(X_datavec_normal,columns=X_columns),
                              pd.DataFrame(y_clst_labels,columns=[Y_names]),
                              how="left",right_index=True,left_index=True)
    XY_datavec = pd.concat([XY_clst_normal,XY_datavec_outlier])
    X_data = XY_datavec.drop(Y_names, axis = 1).values.tolist()
    Y_data = XY_datavec[Y_names].values.tolist()
    #输出每个标签的数量
    print 'Counter:y',Counter(Y_data)
    print'----------------------------------------------'
    #随机采样的少数类，解决类别不平衡问题
    ros = RandomOverSampler(random_state=0)
    #ros = SMOTE(random_state=0)
    X_resampled, Y_resampled = ros.fit_sample(X_data, Y_data)
    print 'Counter:y after using RandomOverSampler',Counter(Y_resampled)
    print'----------------------------------------------'
    if len(X_resampled)>num_used*len(Counter(Y_data)):
        #每个类别分层采样num_used个事例
        x_NoUse_train, X_resampled, y_NoUse_train, Y_resampled = train_test_split(X_resampled, Y_resampled,
                            train_size=None, test_size=num_used*len(Counter(Y_data)),stratify=Y_resampled,random_state=0)
    print 'Counter:used y',Counter(Y_resampled)
    print'----------------------------------------------'

    return X_resampled, Y_resampled

def Sample_unbalanced(X_datavec,Y_datavec,num_used=20000):
    #输出每个标签的数量
    print 'Counter:y',Counter(Y_datavec)
    print'----------------------------------------------'
    #随机采样的少数类，解决类别不平衡问题
    ros = RandomOverSampler(random_state=0)
    #ros = SMOTE(random_state=0)
    X_resampled, Y_resampled = ros.fit_sample(X_datavec, Y_datavec)
    print 'Counter:y after using RandomOverSampler',Counter(Y_resampled)
    print'----------------------------------------------'
    if len(X_resampled)>num_used*len(Counter(Y_datavec)):
        #每个类别分层采样num_used个事例
        x_NoUse_train, X_resampled, y_NoUse_train, Y_resampled = train_test_split(X_resampled, Y_resampled,
                            train_size=None, test_size=num_used*len(Counter(Y_datavec)),stratify=Y_resampled,random_state=0)
    print 'Counter:used y',Counter(Y_resampled)
    print'----------------------------------------------'

    return X_resampled, Y_resampled

# ---------------------------------------------
# 分类算法
# ---------------------------------------------

def GS_LogisticRegression(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'penalty':['l1','l2'], 'C':[0.01,1],
                        'solver':['liblinear']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LogisticRegression(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_penalty = best_par_1['penalty']
    
    tuned_parameters = [{'penalty':[best_par_penalty], 'C':[0.001,0.01,0.2,1,8,50],
                        'solver':['liblinear']}
                        ]
    clf =GridSearchCV(LogisticRegression(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    #y_pred = clf.predict(X_test)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)
    #print "Accuracy score: ",accuracy_score(y_test,y_pred,normalize=True)
    #print "Precision score: ",precision_score(y_test,y_pred)
    #print "Recall score: ",recall_score(y_test,y_pred)

    return clf

def RE_LogisticRegression(X_train, X_test, y_train, y_test, penalty_num = 'l2', C_num= 1):

    clf = LogisticRegression(penalty=penalty_num,C=C_num,solver='sag',n_jobs=-1)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_LinearSVC(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'penalty':['l2'], 'C':[1,0.1],
                        'loss':['hinge','squared_hinge']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LinearSVC(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters = [{'penalty':['l2'], 'C':[0.001,0.01,0.2,1,8,50],
                        'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(LinearSVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_SVC_linear(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'C':[0.001,0.01,0.2,1,8,50],
                        'kernel':['linear']}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf =GridSearchCV(SVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_SVC_rbf(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'C':[1],'kernel':['rbf'],
                        'gamma':[0.001,0.01]}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(SVC(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_gamma = best_par_1['gamma']
    
    tuned_parameters = [{'C':[0.1,1],'kernel':['rbf'],
                        'gamma':[best_par_gamma]}
                        ]
    clf =GridSearchCV(SVC(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_RandomForestClassifier(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[300],
                        'max_features':[0.5,0.8,1]}
                        ]
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(RandomForestClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_f = best_par_1['max_features']
    
    tuned_parameters = [{'n_estimators':[400,500,700],
                        'max_features':[best_par_f]}
                        ]
    clf =GridSearchCV(RandomForestClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_RandomForestClassifier(X_train, X_test, y_train, y_test,n_est = 300, max_f = 0.8):

    clf = RandomForestClassifier(min_samples_split = 20, min_samples_leaf = 8, 
                                 n_estimators = n_est, max_features = max_f,n_jobs=-1)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_GradientBoostingClassifier(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    C_V = StratifiedKFold(n_splits=5,random_state=0)
    best_par_loss = 'deviance'
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_d = best_par_2['max_depth']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':[best_par_loss]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_f = best_par_3['max_features']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)

    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_GradientBoostingClassifier(X_train, X_test, y_train, y_test,n_est= 300, max_f = 0.8):

    clf = GradientBoostingClassifier(min_samples_split = 20, min_samples_leaf = 8,
                                     n_estimators = n_est, max_features = max_f)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

# ---------------------------------------------
# 回归算法
# ---------------------------------------------

def GS_Lasso(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,1,5,10,20,30,50,70]}]
    
    C_V = KFold(n_splits=5,random_state=0)
    clf =GridSearchCV(Lasso(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_Lasso(X_train, X_test, y_train, y_test, alpha_num= 1):

    clf = Lasso(alpha=alpha_num)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_Ridge(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters = [{'alpha':[0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,1,5,10,20,30,50,70]}]
    
    C_V = KFold(n_splits=5,random_state=0)
    clf =GridSearchCV(Ridge(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_Ridge(X_train, X_test, y_train, y_test, alpha_num= 1):

    clf = Ridge(alpha=alpha_num)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_LinearSVR(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'epsilon':[0.06], 'C':[1],
                        'loss':['epsilon_insensitive','squared_epsilon_insensitive']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters_2 = [{'epsilon':[0.06], 'C':[0.001,0.01,0.1,1,10,50],
                        'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_c = best_par_2['C']
    
    tuned_parameters = [{'epsilon':[0.005,0.06,0.1,0.8,5], 'C':[best_par_c],
                        'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(LinearSVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_SVR_linear(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'epsilon':[0.005,0.06],'C':[1],
                        'kernel':['linear']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(SVR(tol = 1e-6), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_eps = best_par_1['epsilon']
    
    tuned_parameters = [{'epsilon':[best_par_eps],'C':[0.1,1],
                        'kernel':['linear']}
                        ]
    clf =GridSearchCV(SVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_SVR_rbf(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    
    tuned_parameters_2 = [{'epsilon':[0.005,0.06],'C':[1],'kernel':['rbf'],
                        'gamma':[0.001]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_2 =GridSearchCV(SVR(tol = 1e-6), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_eps = best_par_2['epsilon']
    
    tuned_parameters = [{'epsilon':[best_par_eps],'C':[0.1,1],'kernel':['rbf'],
                        'gamma':[0.001]}
                        ]
    clf =GridSearchCV(SVR(tol = 1e-6), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_RandomForestRegressor(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    
    tuned_parameters_1 = [{'n_estimators':[300],
                        'max_features':[0.5,0.8,1]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_f = best_par_1['max_features']
    
    tuned_parameters = [{'n_estimators':[400],
                        'max_features':[best_par_f]}
                        ]
    clf =GridSearchCV(RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_RandomForestRegressor(X_train, X_test, y_train, y_test,n_est = 300, max_f= 0.8):

    clf = RandomForestRegressor(min_samples_split = 20, min_samples_leaf = 8, 
                                 n_estimators = n_est, max_features = max_f,n_jobs=-1)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_GradientBoostingRegressor_lslad(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[200],'max_depth':[5],
                        'max_features':[0.5],'loss':['ls','lad']}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_loss = best_par_1['loss']
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':[best_par_loss]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_d = best_par_2['max_depth']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':[best_par_loss]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=-1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_f = best_par_3['max_features']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':[best_par_loss]}
                        ]
    clf =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def GS_GradientBoostingRegressor_huber(*data):
    if len(data)!=4:
        raise NameError('Dimension of the input is not equal to 4')
    X_train, X_test, y_train, y_test = data
    tuned_parameters_1 = [{'n_estimators':[200],'max_depth':[5,10,15,20],
                        'max_features':[0.5],'loss':['huber'],'alpha':[0.9]}
                        ]
    C_V = KFold(n_splits=5,random_state=0)
    clf_1 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_1, cv =C_V, n_jobs=-1)
    clf_1.fit(X_train,y_train)
    best_par_1 = clf_1.best_params_
    best_par_d = best_par_1['max_depth']
    
    tuned_parameters_2 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[0.2,0.5,0.8],'loss':['huber'],'alpha':[0.9]}
                        ]
    clf_2 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_2, cv =C_V, n_jobs=-1)
    clf_2.fit(X_train,y_train)
    best_par_2 = clf_2.best_params_
    best_par_f = best_par_2['max_features']
    
    tuned_parameters_3 = [{'n_estimators':[200],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':['huber'],'alpha':[0.2,0.5,0.9]}
                        ]
    clf_3 =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters_3, cv =C_V, n_jobs=-1)
    clf_3.fit(X_train,y_train)
    best_par_3 = clf_3.best_params_
    best_par_alpha = best_par_3['alpha']
    
    tuned_parameters = [{'n_estimators':[400],'max_depth':[best_par_d],
                        'max_features':[best_par_f],'loss':['huber'],'alpha':[best_par_alpha]}
                        ]
    clf =GridSearchCV(GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8), tuned_parameters, cv =C_V, n_jobs=-1)
    clf.fit(X_train,y_train)
    print "Best parameters set found: ",clf.best_params_
    print "Grid scores: "
    for params_1, mean_score_1, scores_1, in clf_1.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_1, scores_1.std()*2,params_1)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_2, mean_score_2, scores_2, in clf_2.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_2, scores_2.std()*2,params_2)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params_3, mean_score_3, scores_3, in clf_3.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score_3, scores_3.std()*2,params_3)
    print "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-"
    for params, mean_score, scores, in clf.grid_scores_:
        print "\t%0.3f (+/-%0.03f) for %s"%(mean_score, scores.std()*2,params)
    print "optimized score: ",clf.score(X_test,y_test)

    return clf

def RE_GradientBoostingRegressor(X_train, X_test, y_train, y_test,n_est = 300, max_f= 0.8):

    clf = GradientBoostingRegressor(min_samples_split = 20, min_samples_leaf = 8,
                                     n_estimators = n_est, max_features = max_f)
    clf.fit(X_train,y_train)
    print'----------------------------------------------'
    print "Optimized score: ",clf.score(X_test,y_test)

    return clf

# ---------------------------------------------
# 聚类算法
# ---------------------------------------------

def GS_KMeans_parameter(X,options='slow_index'):
    '''
    寻找最优的n_clusters参数
    X：数据样本
    '''
    estimators = [2,3,4,5,6,7,8,9,10,15,20]
    evalue = []
    for num in estimators:
        clst = KMeans(n_clusters=num, init='k-means++', n_jobs=-1)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            if options == 'slow_index':
                #轮廓系数评价指标
                evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            if options == 'fast_index':
                #备用聚类指标，计算这个指标，速度快
                evalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_num_cluster = estimators[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Number of Cluster: %s" % estimators
    print "============================================="
    print "Best Cluster: %s" % best_num_cluster
    print "============================================="

    return best_num_cluster

def Model_KMeans(X,best_num_cluster=8):
    '''
    使用KMeans聚类结果为数据贴标签
    X:数据样本
    '''
    clst = KMeans(n_clusters=best_num_cluster, init='k-means++', n_jobs=-1)
    clst_labels = clst.fit_predict(X)

    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="

    return clst_labels

def GS_DBSCAN_parameter(X,options='slow_index'):
    '''
    利用贪心算法（坐标下降算法），寻找最优的epsilon和min_samples参数
    X：数据样本
    '''
    X = X[0:10000]

    #epsilons = [0.001,0.05,0.06,0.07,0.08,0.1,0.2,0.3,0.4,0.5,0.9,1,2,3,5]
    epsilons = [0.001,0.05,0.1,0.2,0.5,0.7,1,3,5]
    #epsilons = [0.001,0.05,0.1,0.3]
    #min_sample = [1,2,3,4,5,10,15,20,30,50,70,80,100]
    min_sample = [1,2,3,5]
    evalue = []
    mvalue = []
    for epsilon in epsilons:
        clst = DBSCAN(eps = epsilon,n_jobs = -1)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            if options == 'slow_index':
                #轮廓系数评价指标
                evalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            if options == 'fast_index':
                #备用聚类指标，计算这个指标，速度快
                evalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            evalue.append(-1)#为了后面的evalue.index(max(evalue))可以找到正确的eindex而补了一个-1的位置
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_epsilon = epsilons[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Epsilon Value: %s" % epsilons
    print "============================================="
    for num in min_sample:
        clst = DBSCAN(eps = best_epsilon,min_samples = num,n_jobs = -1)
        clst.fit(X)
        if len(set(clst.labels_))>1:
            if options == 'slow_index':
                #轮廓系数评价指标
                mvalue.append(metrics.silhouette_score(X,clst.labels_,metric='euclidean'))
            if options == 'fast_index':
                #备用聚类指标，计算这个指标，速度快
                mvalue.append(metrics.calinski_harabaz_score(X,clst.labels_))
        else :
            mvalue.append(-1)#为了后面的mvalue.index(max(mvalue))可以找到正确的mindex而补了一个-1的位置
    if len(mvalue) == mvalue.count(-1):
        raise NameError('Empty Sequence')
    mindex = mvalue.index(max(mvalue))
    best_num = min_sample[mindex]
    print "Evaluate Ratio: %s" % mvalue
    print "Min Samples Value: %s" % min_sample
    print "============================================="
    print "Best Epsilon: %s" % best_epsilon
    print "Best Min Samples: %s" % best_num
    
    return best_epsilon,best_num

def Model_DBSCAN(X,best_epsilon=0.1,best_num=5,options='slow_index'):
    '''
    使用DBSCAN聚类结果为数据贴标签
    X:数据样本
    ''' 
    clst = DBSCAN(eps = best_epsilon, min_samples = best_num, n_jobs = -1)
    clst.fit(X)
    clst_labels = clst.labels_
    if len(set(clst_labels))>1:
        if options == 'slow_index':
            #轮廓系数评价指标
            evalue=metrics.silhouette_score(X,clst.labels_,metric='euclidean')
        if options == 'fast_index':
            #备用聚类指标，计算这个指标，速度快
            evalue=metrics.calinski_harabaz_score(X,clst.labels_)
    else:
        #小于0即可,-10为了方便
        evalue=-10
    #输出评价系数
    print "============================================="
    print "Evaluate Ratio: %s" % evalue
    print "============================================="
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    
    return clst_labels,evalue

def GS_IsolationForest_parameter(X):
    '''
    寻找最优的contamination参数
    X：数据样本
    '''
    estimators = [-0.01,-0.05,-0.08,-0.1,-0.12,-0.15,-0.17,-0.2,-0.22,-0.25,-0.27,-0.3,-0.35,-0.37,-0.4]
    evalue = []
    data_shape = len(X)
    
    clst = IsolationForest(n_estimators = 100, n_jobs = -1, random_state = 0)
    clst.fit(X)
    scores_pred = clst.decision_function(X)
    #选出异常度最小的值   
    for estimator in estimators:
        #只要evalue中有100了，就说明目前的异常度已经最小，减小重复计算
        if 100 not in evalue:
            contamination_ratio = round(len([i for i in scores_pred if i<estimator])/data_shape,6)
        else :
            contamination_ratio = -1
        if contamination_ratio > 0:
            evalue.append(contamination_ratio)
        else :
            evalue.append(100)
    if len(evalue) == evalue.count(100):
        raise NameError('Empty Sequence')
    eindex = evalue.index(min(evalue))
    best_estimator = estimators[eindex]
    best_contamination = min(evalue)
    print "Contamination Ratio: %s" % evalue
    print "Estimator Value: %s" % [abs(i)+0.5 for i in estimators]
    print "============================================="
    print "Best Estimator: %s" % (abs(best_estimator)+0.5)
    print "Best Contamination: %s" % best_contamination
    print "============================================="
    
    return (abs(best_estimator)+0.5),best_contamination

def Model_IsolationForest(X,best_contamination=0.01):
    '''
    使用Isolation Forest聚类结果为数据贴标签
    X:数据样本
    '''  
    clst = IsolationForest(n_estimators = 100, contamination = best_contamination, n_jobs = -1, random_state = 0)
    clst.fit(X)
    
    scores_pred = clst.decision_function(X)
    clst_labels = clst.predict(X)
    new_scores = [round(abs(i)+0.5,4) for i in scores_pred]
        
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="
    
    return clst_labels,new_scores
  
def GS_LocalOutlierFactor_parameter(X):
    '''
    寻找最优的neighbors和contamination参数
    X：数据样本
    '''
    X = X[0:50000]
    data_shape = len(X)
    neighbors = [15,20,25,30,35,40]
    estimators = [-1,-5,-10,-50,-100,-150,-200,-300,-500,-1000,-10000]
    evalue = []
    mvalue = []
    for neighbor in neighbors:
        clst = LocalOutlierFactor(n_neighbors=neighbor,n_jobs=-1)
        clst.fit(X)
        neighbor_scores= clst.negative_outlier_factor_
        neighbor_contamination_ratio = round(len([i for i in neighbor_scores if i<-1])/data_shape,6)
        if neighbor_contamination_ratio > 0:
            evalue.append(neighbor_contamination_ratio)
        else :
            evalue.append(-1)
    if len(evalue) == evalue.count(-1):
        raise NameError('Empty Sequence')
    eindex = evalue.index(max(evalue))
    best_neighbor = neighbors[eindex]
    print "Evaluate Ratio: %s" % evalue
    print "Neighbor Value: %s" % neighbors
    print "============================================="
    clst = LocalOutlierFactor(n_neighbors=best_neighbor,n_jobs=-1)
    clst.fit(X)
    scores_pred = clst.negative_outlier_factor_
    #选出异常度最小的值   
    for estimator in estimators:
        #只要evalue中有100了，就说明目前的异常度已经最小，减小重复计算
        if 100 not in evalue:
            contamination_ratio = round(len([i for i in scores_pred if i<estimator])/data_shape,6)
        else :
            contamination_ratio = -1
        if contamination_ratio > 0:
            mvalue.append(contamination_ratio)
        else :
            mvalue.append(100)
    if len(mvalue) == mvalue.count(100):
        raise NameError('Empty Sequence')
    mindex = mvalue.index(min(mvalue))
    best_estimator = estimators[mindex]
    best_contamination = min(mvalue)
    print "Contamination Ratio: %s" % mvalue
    print "Estimator Value: %s" % [abs(i) for i in estimators]
    print "============================================="
    print "Best Neighbor: %s" % best_neighbor
    print "Best Estimator: %s" % abs(best_estimator)
    print "Best Contamination: %s" % best_contamination
    print "============================================="
    
    return best_neighbor,best_contamination

def Model_LocalOutlierFactor(X,best_neighbor=20,best_contamination=0.01):
    '''
    使用Local Outlier Factor聚类结果为数据贴标签
    X:数据样本
    '''
    clst = LocalOutlierFactor(n_neighbors = best_neighbor,contamination = best_contamination,n_jobs=-1)
    
    clst_labels = clst.fit_predict(X)
    scores_pred = clst.negative_outlier_factor_
    new_scores = [round(abs(i),4) for i in scores_pred]
        
    for clst_lab in set(clst_labels):
        print "Number of the %s class: %s" % (clst_lab,list(clst_labels).count(clst_lab))
    print "============================================="
    print "Number of the labels: %s" % len(clst_labels)
    print "============================================="
    
    return clst_labels,new_scores

# ---------------------------------------------
# 降维算法
# ---------------------------------------------

def GS_PCA(X):
    '''
    搜索最优PCA降维参数
    dataset：数据样本
    '''
    num0 = 0.999
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = PCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
        if sum_t <= num0:
            ret['99.9%'] = count
    return pca.n_components_,ret

def Model_PCA(X,nums_component):
    '''
    将冗余自由度的数据样本进行降维
    X：数据样本
    nums_component：PCA的降维参数
    '''
    pca = PCA(n_components=nums_component)
    pca.fit(X)
    X_r = pca.transform(X)
    
    return X_r

def GS_IncrementalPCA(X):
    '''
    搜索最优PCA降维参数
    X：数据样本
    '''
    num1 = 0.99
    num2 = 0.98
    num3 = 0.97
    num4 = 0.95
    sum_t = 0
    count = 0
    ret = {}
    pca = IncrementalPCA(n_components=None)
    pca.fit(X)
    ratios = pca.explained_variance_ratio_
    for ratio in ratios:
        sum_t = sum_t + ratio
        count = count + 1
        if sum_t <= num4:
            ret['95%'] = count
        if sum_t <= num3:
            ret['97%'] = count
        if sum_t <= num2:
            ret['98%'] = count
        if sum_t <= num1:
            ret['99%'] = count
    return pca.n_components_, ret

def Model_TSNE(X,nums_component=2):
    '''
    将冗余自由度的数据样本进行降维可视化
    X：数据样本
    nums_component：降维参数
    '''
    X_tsne = TSNE(n_components=nums_component,random_state=33).fit_transform(X)
    
    return X_tsne

