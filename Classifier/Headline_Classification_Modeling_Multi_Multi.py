import pickle
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

with open('Modeling_Data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

model_dict = {
    }
"""'          'LSTM': [LSTM(1000, input_shape=X_train.shape[1:]),
                       250],}

              'LSTM-relu-fat': [LSTM(1000, input_shape=X_train.shape[1:], activation='relu'),
                       250],
              
              'LSTM-relu-skinny': [LSTM(100, input_shape=X_train.shape[1:], activation='relu'),
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
              




for key in model_dict.keys():
    
    dir_ = 'D:/Thesis_Model/Headlines'
    
    model_type = key
    
    if model_type not in os.listdir(dir_):
        os.makedirs(f'{dir_}/{model_type}') 
        
    #Checkpoint
    checkpoint_dir = f'{dir_}/{model_type}/Checkpoint'
    
    checkpoint=ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    
    callbacks = [checkpoint]

    #Model

    model = Sequential()
    
    model.add(model_dict[key][0])
    for i in range(1, len(model_dict[key])):
        model.add(Dense(model_dict[key][i], activation = 'relu'))
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    loss = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam()
    
    model.compile(opt, loss=loss, metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)
    
    hist = model.history
    
    data = {}
    
    for key in hist.history:
        data[key] = hist.history[key]
    
    with open(f'{dir_}/{model_type}/History.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    
