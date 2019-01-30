#!/usr/bin/python
'''
Basic idea is:
    Emebeding the query and document.
    Training a regression model.

Remaining Question:
    How to combine the mean and variance of the score?
    mead as label, vairance as an reverse strength of the label

@Author: Dong Yuan
'''

import re
import os
import numpy as np
import pandas as pd
import tensorflow
from sklearn import metrics, utils
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.layers import Input, Dot, Concatenate
from tensorflow.keras.layers import Conv1D, Embedding
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

MAX_SEQ_LENGTH = 128
MAX_QUERY_LENGTH = 10
VAC_SIZE = 40000
LABELSET_SIZE = 4

def pre_processing():
    t_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    clean = lambda x: re.sub(r'[^\w\s]',' ', re.sub('\d','0', str(x)))
    # for td in [t_data['query'], t_data['product_title'], 
    #         t_data['product_description'], test_data['query'], 
    #         test_data['product_title'], test_data['product_description']]:
    #     td = td.apply(clean)
    t_data['query'] = t_data['query'].apply(clean)
    t_data['product_title'] = t_data['product_title'].apply(clean)
    t_data['product_description'] = t_data['product_description'].apply(clean)
    test_data['query'] = test_data['query'].apply(clean)
    test_data['product_title'] = test_data['product_title'].apply(clean)
    test_data['product_description'] = test_data['product_description'].apply(clean)
    token = text.Tokenizer()
    whole_text = t_data['query'].append(t_data['product_title'])
    whole_text = whole_text.append(t_data['product_description']) #.map(str).apply(lambda x: x.encode('utf-8')))
    token.fit_on_texts(whole_text.values)
    word_index = token.word_index

    # Treat no difference between title and description
    t_data['product_title'] = t_data['product_title'] + " "
    t_data['product_title'] = t_data['product_description']
    test_data['product_title'] = test_data['product_title'] + " "
    test_data['product_title'] = test_data['product_description']

    split_ind = int(t_data.shape[0]*0.999)
    train_data = t_data[:split_ind]
    val_data = t_data[split_ind:]
    train_query = sequence.pad_sequences(
            token.texts_to_sequences(train_data['query']), maxlen=10)
    train_ptitle = sequence.pad_sequences(
            token.texts_to_sequences(train_data['product_title']), maxlen=128)
    # train_pdescription = sequence.pad_sequences(
    #         token.texts_to_sequences(train_data['product_description']), maxlen=80)
    val_query = sequence.pad_sequences(
            token.texts_to_sequences(val_data['query']), maxlen=10)
    val_ptitle = sequence.pad_sequences(
            token.texts_to_sequences(val_data['product_title']), maxlen=128)
    # train_pdescription = sequence.pad_sequences(
    #         token.texts_to_sequences(train_data['product_description']), maxlen=80)
    test_query = sequence.pad_sequences(
            token.texts_to_sequences(test_data['query']), maxlen=10)
    test_ptitle = sequence.pad_sequences(
            token.texts_to_sequences(test_data['product_title']), maxlen=128)
    # train_pdescription = sequence.pad_sequences(
    #         token.texts_to_sequences(train_data['product_description']), maxlen=80)

    def cal_label(x, y):
        return x

    def transform_label(x):
        return x-1
        return np.eye(LABELSET_SIZE, dtype=int)[x-1]

    ret_train = {
                'queries': train_query, 
                'ptitles': train_ptitle, 
                'labels': train_data['median_relevance'].apply(transform_label)
                }
    ret_val = {
                'queries': val_query, 
                'ptitles': val_ptitle, 
                'labels': val_data['median_relevance'].apply(transform_label)
                }
    ret_test = {
                'queries': test_query, 
                'ptitles': test_ptitle, 
                'labels': ""
                }
    return (ret_train, ret_val, ret_test)

def model():
    query = Input(shape=(MAX_QUERY_LENGTH,))
    embedding_q = Embedding(VAC_SIZE, 64, input_length=MAX_QUERY_LENGTH, 
                            trainable=False)(query)
    conv_q= Conv1D(100, 2, padding='same', activation='relu')(embedding_q)
    conv_q = Dropout(0.25)(conv_q)
    pool_q = GlobalMaxPooling1D()(conv_q)

    title = Input(shape=(MAX_SEQ_LENGTH, ))
    embedding_a  = Embedding(VAC_SIZE, 64, input_length=MAX_QUERY_LENGTH,
                             trainable=False)(title)
    conv_a = Conv1D(100, 4, padding='same', activation='relu')(embedding_a)
    conv_a = Dropout(0.25)(conv_a)
    pool_a = GlobalMaxPooling1D()(conv_a)

    sim = Dot(-1)([Dense(100, use_bias=False)(pool_q), pool_a])
    model_sim = Concatenate()([pool_q, pool_a, sim])
    print(model_sim)

    model_final = Dropout(0.5)(model_sim)
    model_final = Dense(201)(model_final)
    model_final = Dropout(0.5)(model_final)
    print(model_final)
    model_final = Dense(LABELSET_SIZE, activation='softmax')(model_final)
    print(model_final)

    model = Model(inputs=[query, title], outputs=model_final)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(opt, loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    print(model.summary())
    return model

def main():
    batch_size = 32
    training_epoch = 15
    train_data, val_data, test_data = pre_processing()
    print(train_data['labels'])
    cmodel = model()
    cmodel.fit([train_data['queries'], train_data['ptitles']], 
               train_data['labels'],
               batch_size=batch_size,
               epochs=training_epoch,
               validation_split=0.2,
               verbose=1)
    # for e in range(training_epoch):
    #     train_data['queris'], train_data['ptitles'], train_data['labels'] = \
    #         utils.shuffle(train_data['queris'], train_data['ptitles'], 
    #                       train_data['labels'])

if __name__ == '__main__':
    main()
    print('Running...')
