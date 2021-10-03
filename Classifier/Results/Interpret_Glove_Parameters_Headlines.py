import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

dir_ = 'D:/Thesis_Model/Headlines'
import tensorflow.keras.backend as K
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Embedding, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dir_ = 'D:/Thesis_Model/Headlines - Non Pretrained 2'
model_loc = 'D:/Thesis_Model'

max_length = 50
#num_w = 85000
vec_length = 300
#vec_length = 24

class_ = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'

with open(f'{class_}/Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

with open(f'{model_loc}/Token_Data.pkl', 'rb') as f:
    tok, embedding_matrix = pickle.load(f)

num_w = len(tok.word_index) + 1
#num_w = 15000


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


#

model_dict = {'LSTM': [LSTM(1000, input_shape=X_train.shape[1:]),
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

results_dict = {}
failed_list = []
for key in model_dict.keys():
    try:
        model_type = key
        
        checkpoint_dir = f'{dir_}/{model_type}/Checkpoint'
        
        ###Model
        
        model = Sequential()
        
        #model.add(Embedding(num_w, vec_length, weights=[embedding_matrix], input_length=max_length, trainable=False))
        model.add(Embedding(num_w, vec_length, input_length=max_length))
        
        model.add(model_dict[key][0])
        
        for i in range(1, len(model_dict[key])):
            model.add(Dense(model_dict[key][i], activation = 'relu'))
        
        model.add(Dense(1, activation='sigmoid'))
        
        loss = keras.losses.BinaryCrossentropy()
        opt = keras.optimizers.Adam()
        
        model.compile(opt, loss=loss, metrics=['accuracy'])
        
        model.load_weights(checkpoint_dir)
        
        pred = model.predict_classes(X_test)
        
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        
        
        results_dict[key] = {}
        
        results_dict[key]['Parameters'] = non_trainable_count
        results_dict[key]['Trainable'] = trainable_count
    except:
        failed_list.append(key)


df = pd.DataFrame(results_dict)

df.to_csv('Parameters_Headline_NoNGlove.csv')
