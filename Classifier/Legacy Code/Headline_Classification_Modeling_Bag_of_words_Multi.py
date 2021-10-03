import pickle
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Embedding, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dir_ = 'D:/Thesis_Model/Headlines - Non Pretrained_2'

max_length = 50
num_w = 15000

with open('Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

X_train = training[:, 0]
y_train = training[:, 1]

X_test = testing[:, 0]
y_test = testing[:, 1]


###
tok = Tokenizer(num_words=num_w)
tok.fit_on_texts(X_train)

train = tok.texts_to_sequences(X_train)
test = tok.texts_to_sequences(X_test)

X_train = pad_sequences(train, padding='post', truncating = 'post', maxlen=max_length)
X_test = pad_sequences(test, padding='post', truncating = 'post', maxlen=max_length)

y_train = y_train.astype(int)
y_test = y_test.astype(int)


with open(f'{dir_}/Token.pkl', 'wb') as f:
    pickle.dump(tok, f)

###
model_dict = {
    'BERT':[]
    }

"""'LSTM': [LSTM(1000, input_shape=X_train.shape[1:]),
                       250],
              'LSTM-relu-fat': [LSTM(1000, input_shape=X_train.shape[1:]),
                       250],
              
              'LSTM-relu-skinny': [LSTM(100, input_shape=X_train.shape[1:]),
                      25],

              'LSTM-relu-tall': [LSTM(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250, 50],
              
              'RNN': [SimpleRNN(1000, input_shape=X_train.shape[1:]),
                       250],
              
              'RNN-relu-fat': [SimpleRNN(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250],
              
              'RNN-relu-skinny': [SimpleRNN(100, input_shape=X_train.shape[1:], activation='relu'),
                       25],
              
              'RNN-relu-tall': [SimpleRNN(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250, 50],
               
              'GRU': [GRU(1000, input_shape=X_train.shape[1:]),
                       250],
              
              'GRU-relu-fat':[GRU(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250],
              
              'GRU-relu-skinny':[GRU(100, input_shape=X_train.shape[1:], activation='relu'),
                       25],
              
              'GRU-relu-tall':[GRU(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250, 50],
              
              'BILSTM': [Bidirectional(LSTM(1000, input_shape=X_train.shape[1:])),
                       250],
              
              'BILSTM-relu-fat': [Bidirectional(LSTM(1000, input_shape=X_train.shape[1:], activation='relu')),
                       250],
              
              'BILSTM-relu-skinny': [Bidirectional(LSTM(100, input_shape=X_train.shape[1:], activation='relu')),
                       25],
              
              'BILSTM-relu-tall': [Bidirectional(LSTM(1000, input_shape=X_train.shape[1:], activation='relu')),
                       250, 50],
              
              'AveragePooling':[keras.layers.GlobalAveragePooling1D(),
                       250],
              
              'MaxPooling':[keras.layers.GlobalMaxPooling1D(),
                            250],
              }
"""
###

for key in model_dict.keys():
    model_type = key
    if model_type not in os.listdir(dir_):
        os.makedirs(f'{dir_}/{model_type}')
    
    checkpoint_dir = f'{dir_}/{model_type}/Checkpoint'
    
    checkpoint=ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    
    callbacks = [checkpoint]
    
    ###Model
    
    model = Sequential()
    
    model.add(Embedding(num_w, 24, input_length=max_length))
    
    model.add(model_dict[key][0])
    
    for i in range(1, len(model_dict[key])):
        model.add(Dense(model_dict[key][i], activation = 'relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    
    loss = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam()
    
    model.compile(opt, loss=loss, metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)
    
    #Results
    hist = model.history
    
    data = {}
    
    for key in hist.history:
        data[key] = hist.history[key]
    
    with open(f'{dir_}/{model_type}/History.pkl', 'wb') as f:
        pickle.dump(data, f)

