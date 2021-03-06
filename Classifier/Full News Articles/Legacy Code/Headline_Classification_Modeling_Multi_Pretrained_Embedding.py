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

dir_ = 'D:/Thesis_Model/Full_Article_Adjusted'
model_loc = 'D:/Thesis_Model'

max_length = 1400
#num_w = 85000
vec_length = 300

with open(f'{model_loc}/Word_Arrays_Article_Altered.pkl', 'rb') as f:
    training, testing = pickle.load(f)

with open(f'{model_loc}/Token_Data_Article.pkl', 'rb') as f:
    tok, embedding_matrix = pickle.load(f)

num_w = len(tok.word_index) + 1

X_train = training[:, 0]
y_train = training[:, 1]

X_test = testing[:, 0]
y_test = testing[:, 1]


###

train = tok.texts_to_sequences(X_train)
test = tok.texts_to_sequences(X_test)

X_train = pad_sequences(train, padding='pre', truncating = 'pre', maxlen=max_length)
X_test = pad_sequences(test, padding='pre', truncating = 'pre', maxlen=max_length)

y_train = y_train.astype(int)
y_test = y_test.astype(int)


###
model_dict = {'LSTM':[LSTM(500, input_shape=X_train.shape[1:]),
                       150],
              
              'RNN': [SimpleRNN(500, input_shape=X_train.shape[1:]),
                       150],
              
              'GRU': [GRU(500, input_shape=X_train.shape[1:]),
                       150],
              
              'BILSTM': [Bidirectional(LSTM(200, input_shape=X_train.shape[1:])),
                       50],
              
              'CNN':[Conv1D(1000, 8),
                     MaxPooling1D(),
                     100,
                     50],
              'CNN_LSTM':[Conv1D(1000, 8),
                     LSTM(500),
                     100,
                     50]
              }

"""
'LSTM':[LSTM(500, input_shape=X_train.shape[1:]),
                       150],
              'RNN': [SimpleRNN(500, input_shape=X_train.shape[1:]),
                       150],
              'GRU': [GRU(500, input_shape=X_train.shape[1:]),
                       150],
              'BILSTM': [Bidirectional(LSTM(200, input_shape=X_train.shape[1:])),
                       50],
              'AveragePooling':[keras.layers.GlobalAveragePooling1D(),
                       150],
              'MaxPooling':[keras.layers.GlobalMaxPooling1D(),
                            150],
"""

"""
'LSTM': [LSTM(500, input_shape=X_train.shape[1:]),
                       150],
              
              'LSTM-relu-fat': [LSTM(500, input_shape=X_train.shape[1:], activation='relu'),
                       150],
              
              'LSTM-relu-skinny': [LSTM(100, input_shape=X_train.shape[1:], activation='relu'),
                      25],

              'LSTM-relu-tall': [LSTM(500, input_shape=X_train.shape[1:], activation='relu'),
                       150, 50],
              
              'RNN': [SimpleRNN(500, input_shape=X_train.shape[1:]),
                       150],
              
              'RNN-relu-fat': [SimpleRNN(500, input_shape=X_train.shape[1:], activation='relu'),
                       150],
              
              'RNN-relu-skinny': [SimpleRNN(100, input_shape=X_train.shape[1:], activation='relu'),
                       25],
              
              'RNN-relu-tall': [SimpleRNN(500, input_shape=X_train.shape[1:], activation='relu'),
                       150, 50],
               
              'GRU': [GRU(500, input_shape=X_train.shape[1:]),
                       150],
              
              'GRU-relu-fat':[GRU(500, input_shape=X_train.shape[1:], activation='relu'),
                       150],
              
              'GRU-relu-skinny':[GRU(100, input_shape=X_train.shape[1:], activation='relu'),
                       25],
              
              'GRU-relu-tall':[GRU(500, input_shape=X_train.shape[1:], activation='relu'),
                       150, 50],
              
              'BILSTM': [Bidirectional(LSTM(500, input_shape=X_train.shape[1:])),
                       150],
              
              'BILSTM-relu-fat': [Bidirectional(LSTM(500, input_shape=X_train.shape[1:], activation='relu')),
                       150],
              
              'BILSTM-relu-skinny': [Bidirectional(LSTM(100, input_shape=X_train.shape[1:], activation='relu')),
                       25],
              
              'BILSTM-relu-tall': [Bidirectional(LSTM(500, input_shape=X_train.shape[1:], activation='relu')),
                       150, 50],
              
              'AveragePooling':[keras.layers.GlobalAveragePooling1D(),
                       150],
              
              'MaxPooling':[keras.layers.GlobalMaxPooling1D(),
                            150],
              }
"""
###

for key in model_dict.keys():
    model_type = key
    if model_type not in os.listdir(dir_):
        os.makedirs(f'{dir_}/{model_type}')
    
    checkpoint_dir = f'{dir_}/{model_type}/Checkpoint'
    
    checkpoint=ModelCheckpoint(checkpoint_dir, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
    early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0, patience=10, verbose=0,
                    mode='auto', baseline=None, restore_best_weights=False
                )
    
    callbacks = [checkpoint, early_stopping]
    
    ###Model
    model = Sequential()
    
    model.add(Embedding(num_w, vec_length, weights=[embedding_matrix], input_length=max_length, trainable=False))
    
    for i in range(len(model_dict[key])):
        if type(model_dict[key][i]) == int:
            model.add(Dense(model_dict[key][i], activation = 'relu'))
        else:
            model.add(model_dict[key][i])
    """
    model.add(model_dict[key][0])
    
    for i in range(1, len(model_dict[key])):
        model.add(Dense(model_dict[key][i], activation = 'relu'))
    """
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
    
    del checkpoint
    del callbacks
