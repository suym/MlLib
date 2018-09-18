#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
DL_Package.py
深度学习工具包
'''
from __future__ import division

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras import regularizers
from keras import backend as K
from keras import metrics as keras_metrics
from keras.layers import Lambda as keras_Lambda
from keras.models import Sequential, Model, load_model  
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD

# ---------------------------------------------
# 神经网络算法
# ---------------------------------------------

def make_cla_MFNN(data_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=data_dim,kernel_initializer='normal',activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='softmax'))
    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

    return model

def GS_cla_MFNN(X_train, X_test, y_train, y_test,epoch=10):

    X_dim = X_train.shape[1]
    
    model = make_cla_MFNN(X_dim)
    model.fit(X_train, y_train,
          epochs=epoch,
          batch_size=128)

    score, acc = model.evaluate(X_test, y_test, batch_size=128)
    print "\nTest score: %.3f, accuracy: %.3f" % (score, acc)
    
    return model

# ---------------------------------------------
# 异常检测算法
# ---------------------------------------------

def GS_auto_encoder_parameter(X,pca_dim=30):
    '''
    利用贪心算法（坐标下降算法），寻找单隐层自编码器最优的encoding_dim参数
    PS: loss='mean_squared_error'是因为dataset中数据中的值不是0或1
    '''

    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    if pca_dim > 3:
        encoding_dim = [pca_dim-3,pca_dim-1,pca_dim,pca_dim+1,pca_dim+3]
    else:
        encoding_dim = [pca_dim,pca_dim+1,pca_dim+3]
    input_img = Input(shape=(dim_data,))
    score =[]
    
    for en_dim in encoding_dim:
        encoded = Dense(en_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
        #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
        decoded = Dense(dim_data, activation='sigmoid')(encoded)
        
        autoencoder = Model(input=input_img, output=decoded)
        encoder = Model(input=input_img, output=encoded)
        
        encoded_input = Input(shape=(en_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        
        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
        autoencoder.fit(X_train, X_train,
                epochs=80,
                batch_size=64)
        score.append(autoencoder.evaluate(X_test, X_test, batch_size=64))
        
    sindex = score.index(min(score))
    best_encoding_dim = encoding_dim[sindex]
    print "Evaluate Ratio: %s" % score
    print "Encoding_dim Value: %s" % encoding_dim
    print "============================================="
    print "Best encoding_dim: %s" % best_encoding_dim
    
    return best_encoding_dim
    
def Model_auto_encoder(X,best_encoding_dim):
    '''
    建立单隐层的自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    dim_data = X.shape[1]
    en_dim = best_encoding_dim
    input_img = Input(shape=(dim_data,))
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(en_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(encoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
        
    encoded_input = Input(shape=(en_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    #如何选择loss函数 binary_crossentropy mean_squared_error mean_absolute_error
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X, X,
                epochs=80,
                batch_size=64)
    encoded_imgs = encoder.predict(X)
    X_decoded = decoder.predict(encoded_imgs)
    
    return X_decoded
    
def Model_deep_auto_encoder(X):
    '''
    建立多隐层的自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    input_img = Input(shape=(dim_data,))
    batch_size = 64
    epochs = 4
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(decoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')#mean_absolute_error
    autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                )
    loss = autoencoder.evaluate(X_test, X_test, batch_size=batch_size)
    X_decoded = autoencoder.predict(X)
    X_decoded = np.array(X_decoded)
    X_tem = np.power(abs(X-X_decoded),2)
    X_diff_loss = [sum(i)/dim_data for i in X_tem]
    print "============================================="
    print "Test Loss: %s" % loss
    print "============================================="
    
    return loss,X_diff_loss

def GS_deep_auto_encoder_parameter(loss,X_diff_loss):
    '''
    根据loss值和X_diff_loss，寻找最优的异常度分界线
    '''
    X = [i/loss*1.0 for i in X_diff_loss]
    X_shape = len(X)
    evalue = []
    #1.0是为了找出大于loss值
    estimators = [1.0,1.05,1.2,1.5,1.6,1.7,1.8,2,3]
    #选出异常度最小的值   
    for estimator in estimators:
        contamination_ratio = round(len([i for i in X if i>estimator])/X_shape,6)
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
    print "Estimator Value: %s" % estimators
    print "============================================="
    print "Best Estimator: %s" % best_estimator
    print "Best Contamination: %s" % best_contamination
    print "============================================="
    
    new_X  = []
    for index in X:
        if index > best_estimator:
            new_X.append(-1)
        else :
            new_X.append(1)
    
    return new_X

def Model_deep_auto_encoder_noisy(X):
    '''
    建立去噪多隐层的自编码机
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    dim_data = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    noise_factor = 0.01
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 
    #因为dataset中的数据在预处理中已经被归一化到0到1之间，所以为了之后的sigmoid层(输出为0到1之间)，即使加了噪声，值也要在0到1之间
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    input_img = Input(shape=(dim_data,))
    batch_size = 64
    epochs = 80
    #activity_regularizer=regularizers.l1(10e-5),对隐层单元施加稀疏性约束的话，会得到更为紧凑的表达
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoded = Dense(dim_data, activation='sigmoid')(decoded)
        
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(X_train_noisy, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test_noisy, X_test))
    X_decoded = autoencoder.predict(X)
    
    return X_decoded
    
def Model_variational_autoencoder(X):
    '''
    变分自编码器
    使用详情：https://github.com/MoyanZitto/keras-cn/blob/master/docs/legacy/blog/autoencoder.md 
    '''
    original_dim = X.shape[1]
    X_train, X_test = train_test_split(X,train_size=0.75, test_size=0.25,random_state=0)
    
    batch_size = 64
    latent_dim = 30
    intermediate_dim = 128
    epochs = 80
    epsilon_std = 1.0
    # 建立编码网络，将输入影射为隐分布的参数
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    # 从这些参数确定的分布中采样，这个样本相当于之前的隐层值
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    z = keras_Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # 采样得到的点映射回去重构原输入
    decoder_h = Dense(intermediate_dim, activation='relu')
    #因为输入的数据取值范围是0到1，所以用sigmoid可以很好的对应
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    # 构建VAE模型
    vae = Model(x, x_decoded_mean)
    # 使用端到端的模型训练，损失函数是一项重构误差，和一项KL距离
    #xent_loss = original_dim * keras_metrics.binary_crossentropy(x, x_decoded_mean)
    xent_loss = original_dim * keras_metrics.mean_squared_error(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    
    vae.fit(X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, None))
    
    X_encoded = vae.predict(X)
    
    return X_encoded

