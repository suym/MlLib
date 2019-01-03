#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Log_Pattern_seq2seq.py
用于从原始日志中抽取出结构化pattern的算法模型
'''

__author__ = "Su Yumo <suyumo@buaa.edu.cn>"

import re
import os
import operator
import sys
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences


def parse_train_data(dir_of_inputdata):
    X_dataset = []
    Y_dataset = []
    X_tem = []
    Y_tem = []
    max_encoder_seq_length = 0
    max_decoder_seq_length = 0
    regEx = re.compile('\\W*')
    with open(dir_of_inputdata) as f:
        for line in f.readlines():
            data = json.loads(line)
            for data_str in data:
                #获得raw_event,pattern
                X_tem.append(data_str['key'])
                Y_tem.append(data_str['value'])

    for line in X_tem:
        #将句子分词
        listoftoken = regEx.split(line)
        #去掉空格并将字符转为小写
        tem = [tok.lower() for tok in listoftoken if len(tok)>0]
        #将句子中的数字转化为特定的字符isdigit
        tem_digit = ['isdigit' if x.isdigit() else str(x) for x in tem]
        #获得最长句子的单词数
        if len(tem_digit)>max_encoder_seq_length:
            max_encoder_seq_length = len(tem_digit)
        #将一个list中的字符拼接成字符串，不再是list
        tem_isdigit = ' '.join(tem_digit)
        X_dataset.append(tem_isdigit)

    for line in Y_tem:
        #将句子分词
        listoftoken = regEx.split(line)
        #去掉空格并将字符转为小写
        tem = [tok.lower() for tok in listoftoken if len(tok)>0]
        #将句子中的数字转化为特定的字符isdigit
        tem_digit = ['isdigit' if x.isdigit() else str(x) for x in tem]
        #获得最长句子的单词数
        if len(tem_digit)>max_decoder_seq_length:
            max_decoder_seq_length = len(tem_digit)
        #将一个list中的字符拼接成字符串，不再是list
        tem_isdigit = ' '.join(tem_digit)
        #补上<start>和<end>标记
        tem_isdigit = '<-^-start-^->' + ' ' + tem_isdigit + ' ' + '<-^-end-^->'
        Y_dataset.append(tem_isdigit)

    print "Max sequence length for inputs: %s"%max_encoder_seq_length
    print "Max sequence length for outputs: %s"%max_decoder_seq_length
    print'----------------------------------------------'

    Max_Sequences_Length = max(max_encoder_seq_length,max_decoder_seq_length+2)

    return X_dataset, Y_dataset, Max_Sequences_Length

def _to_word_index(X_dataset,word_index,Max_Sequences_Length):
    #将所有的词编码为数字
    x_data = []
    for words in X_dataset:
        x_tem = []
        for word in words.split():
            if word in word_index:
                x_tem.append(word_index[word])
            else:
                x_tem.append(word_index['UNK'])
        x_data.append(x_tem)
    #将长度不足Max_Sequences_Length的句子用0填充,大于Max_Sequences_Length句子被截断,'post'代表向后填充和截取
    x_data = pad_sequences(x_data, maxlen=Max_Sequences_Length, padding='post', truncating='post')
    x_data = np.array(x_data)

    return x_data 

def to_word_index(X_dataset,Y_dataset,Max_Sequences_Length):
    #得到所有word的词频，X_dataset形式为['a s d','as df sd'],Y_dataset形式为['<-^-start-^-> as <-^-end-^->',]
    word_freqs = Counter()
    for words in X_dataset:
        for word in words.split():
            word_freqs[word] = word_freqs[word] +1
    for words in Y_dataset:
        for word in words.split():
            word_freqs[word] = word_freqs[word] +1 
    #得到word to index，i从2开始，word_freqs.most_common返回的是形如('',12)，因此用x[0]
    word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(len(word_freqs)-1))}
    #将所有的未知字符都归为'UNK'
    word_index["UNK"] = 1

    x_data = _to_word_index(X_dataset,word_index,Max_Sequences_Length)
    y_input_data = _to_word_index(Y_dataset,word_index,Max_Sequences_Length)
    vocab_size = len(word_index)
    #`y_input_data`慢`y_target_data`一个时间步
    y_target_data = to_one_hot(y_input_data,Max_Sequences_Length,vocab_size)
    print 'size of word_index: %s'%vocab_size
    print'----------------------------------------------'
    print 'word to index:'
    print sorted(word_index.items(),key = lambda x:x[1],reverse = False)
    print'----------------------------------------------'

    return x_data, y_input_data, y_target_data, word_index, vocab_size

def to_one_hot(y_input_data,Max_Sequences_Length,vocab_size):
    y_target_data = np.zeros((len(y_input_data), Max_Sequences_Length, vocab_size), dtype=np.float32)
    for i in range(len(y_input_data)):
        #y_target_data做one-hot编码，且偏移一位
        for t, index in enumerate(y_input_data[i, 1:]):
            y_target_data[i, t, index] = 1.0
        #补上最后一个数为0即为空(由于y_input_data[i, 1:]造成)
        y_target_data[i, -1, 0] = 1.0

    return y_target_data

def build_basic_model(latent_dim,vocab_size):
    #定义一个编码器
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    #编码器和解码器共用一个词嵌入层
    embedding = Embedding(vocab_size+1, latent_dim, name='embedding')
    x = embedding(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,
                    name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(x)
    encoder_states = [state_h, state_c]

    # 定义一个解码器, 用'encoder_states'作为初始值.
    decoder_inputs = Input(shape=(None,),name='decoder_inputs')
    y = embedding(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,
                        name='decoder_lstm')
    lstm_outputs, _, _ = decoder_lstm(y,
                                     initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(lstm_outputs)

    #定义一个训练模型，输入`encoder_input_data`和`decoder_input_data`，输出为`decoder_target_data`
    basic_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    #当预测目标是整形数字的时候可以用sparse_categorical_crossentropy
    basic_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    basic_model.summary()

    return basic_model

def trian_model(*data):
    x_data,y_input_data,y_target_data,batch_size,epochs,latent_dim,vocab_size = data
    basic_model = build_basic_model(latent_dim,vocab_size)
    #`decoder_input_data`慢`decoder_target_data`一个时间步
    basic_model.fit([x_data,y_input_data], y_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

    return basic_model 

def save_model(basic_model):
    if not os.path.isfile('./seq2seq_model.h5'):
        basic_model.save('seq2seq_model.h5')    

def build_inference_model(basic_model,latent_dim):
    #得到一个推断编码器
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = basic_model.get_layer('embedding')(encoder_inputs)
    _, state_h, state_c= basic_model.get_layer('encoder_lstm')(encoder_embedding)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    #定义推断解码器的输入
    decoder_inputs = Input(shape=(None,))
    decoder_state_h = Input(shape=(latent_dim,))
    decoder_state_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_h, decoder_state_c]

    #得到一个推断解码器
    decoder_embedding = basic_model.get_layer('embedding')(decoder_inputs)
    lstm_outputs, de_state_h, de_state_c = basic_model.get_layer('decoder_lstm')(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [de_state_h, de_state_c]
    decoder_outputs = basic_model.get_layer('decoder_dense')(lstm_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)

    return encoder_model, decoder_model

def decode_sequence(input_seq, word_index, Max_Sequences_Length, encoder_model, decoder_model):
    #下标对文本的索引表
    reverse_word_index = {v:k for k, v in word_index.items()}
    #得到编码器的输出状态
    states_value = encoder_model.predict(input_seq)

    #创造一个空的目标序列
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['<-^-start-^->']
    
    #进行句子的恢复
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output, de_state_h, de_state_c = decoder_model.predict([target_seq] + states_value)
        decoder_states = [de_state_h, de_state_c]

        sampled_token_index = np.argmax(output[0, -1, :])
        sampled_word = reverse_word_index[sampled_token_index]
        decoded_sentence =  decoded_sentence + sampled_word + ' '
        
        if sampled_word == '<-^-end-^->' or len(decoded_sentence) > Max_Sequences_Length:
            stop_condition = True
            
        #更新target_seq
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        #更新状态
        states_value = decoder_states
        
    return decoded_sentence

def predict_data(x_data, word_index, Max_Sequences_Length, encoder_model, decoder_model):
    for seq_index in range(len(x_data)):
        input_seq = x_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, word_index, Max_Sequences_Length, encoder_model, decoder_model)
        print 'Input sentence:'
        print x_data[seq_index]
        print 'Decoded sentence:' 
        print decoded_sentence
        print '----------------------------------------------'

def storePara(dir_of_storePara,word_index,Max_Sequences_Length):
    para_dict = {}
    para_dict['word_index']=word_index
    para_dict['Max_Sequences_Length'] = Max_Sequences_Length
    with open(dir_of_storePara,'w') as f:
        json.dump(para_dict,f)

def loadModel():
    basic_model = load_model('seq2seq_model.h5')

    return basic_model

def Duration(seconds):
    seconds = long(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365.242199)

    minutes = long(minutes)
    hours = long(hours)
    days = long(days)
    years = long(years)

    duration = []
    if years > 0:
        duration.append('%d year' % years + 's'*(years != 1))
    else:
        if days > 0:
            duration.append('%d day' % days + 's'*(days != 1))
        if hours > 0:
            duration.append('%d hour' % hours + 's'*(hours != 1))
        if minutes > 0:
            duration.append('%d minute' % minutes + 's'*(minutes != 1))
        if seconds > 0:
            duration.append('%d second' % seconds + 's'*(seconds != 1))

    return ' '.join(duration)

def main():
    #静默弃用sklearn警告
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    options = 'train'
    dir_of_inputdata = './data/final_data'
    dir_of_outputdata = './data/outdata'
    batch_size =256
    epochs = 10
    latent_dim = 128

    if options == 'train':
        time_start = time()
        dir_of_storePara = './data/Parameters.json'
        X_dataset, Y_dataset, Max_Sequences_Length = parse_train_data(dir_of_inputdata)
        x_data, y_input_data, y_target_data, word_index, vocab_size = to_word_index(X_dataset,Y_dataset,Max_Sequences_Length)
        print'--------------Train data shape----------------'
        print 'x_data.shape:',x_data.shape
        print'----------------------------------------------'
        print 'y_input_data.shape:',y_input_data.shape
        print'----------------------------------------------'
        print 'y_target_data.shape:',y_target_data.shape
        print'----------------------------------------------'
        x_train,x_test,y_input_train,y_input_test,y_target_train,y_target_test = train_test_split(x_data,y_input_data,y_target_data,
                                                        train_size=None, test_size=20, random_state=0)
        basic_model = trian_model(x_train,y_input_train,y_target_train,batch_size,epochs,latent_dim,vocab_size)
        save_model(basic_model)
        encoder_model, decoder_model = build_inference_model(basic_model,latent_dim)
        predict_data(x_test, word_index, Max_Sequences_Length, encoder_model, decoder_model)
        storePara(dir_of_storePara, word_index, Max_Sequences_Length)
        duration = Duration(time()-time_start)
        print 'Total run time: %s'%duration


if __name__ == '__main__':
    main()