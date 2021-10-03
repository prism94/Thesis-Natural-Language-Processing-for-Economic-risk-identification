import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

dir_ = 'D:/Thesis_Model/Headlines'


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Embedding, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dir_ = 'D:/Thesis_Model/Headlines'

class_dir = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'


with open(f'{class_dir}/Modeling_Data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)


model_dict = {'LSTM': [LSTM(1000, input_shape=X_train.shape[1:]),
                       250],
              
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

results_dict = {}

for key in model_dict.keys():
    model_type = key
    
    checkpoint_dir = f'{dir_}/{model_type}/Checkpoint'
    
    ###Model
    
    model = Sequential()
    
    model.add(model_dict[key][0])
    
    for i in range(1, len(model_dict[key])):
        model.add(Dense(model_dict[key][i], activation = 'relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    loss = keras.losses.BinaryCrossentropy()
    opt = keras.optimizers.Adam()
    
    model.compile(opt, loss=loss, metrics=['accuracy'])
    
    model.load_weights(checkpoint_dir)
    
    pred = model.predict_classes(X_test)
    
    results_dict[key] = {}
    
    results_dict[key]['Accuracy'] = accuracy_score(y_test, pred)
    results_dict[key]['Precision'] =  precision_score(y_test, pred)
    results_dict[key]['Recall'] = recall_score(y_test, pred)
    results_dict[key]['F1_score'] = f1_score(y_test, pred)

df = pd.DataFrame(results_dict)

df.to_csv('Model_Results_Pre-trained.csv')
