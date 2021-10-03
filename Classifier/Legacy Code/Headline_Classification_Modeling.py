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

dir_ = 'D:/Thesis_Model/headlines'

model_type = 'biLSTM'

params = 'adam-relu-skinny'

model_type = model_type + '-' + params

if model_type not in os.listdir(dir_):
    os.makedirs(f'{dir_}/{model_type}') 

#Checkpoint
checkpoint_dir = f'D:/Thesis_Model/{model_type}/Checkpoint'

checkpoint=ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

callbacks = [checkpoint]



#LSTM

model = Sequential()

model.add(Bidirectional(LSTM(1000, input_shape=X_train.shape[1:], activation='relu')))
model.add(Dense(250, activation = 'relu'))
#model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam()

model.compile(opt, loss=loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)



"""
#RNN

model = Sequential()

model.add(SimpleRNN(250, input_shape=X_train.shape[1:], activation = keras.layers.LeakyReLU()))
#model.add(Dropout(0.25))
#model.add(Dense(500, activation =  keras.layers.LeakyReLU()))
model.add(Dense(50, activation =  keras.layers.LeakyReLU()))
model.add(Dense(1, activation = 'sigmoid'))

loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam()

model.compile(opt, loss=loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

"""
"""

#CNN
model = Sequential()

model.add(Conv1D(500, kernel_size=5, input_shape=X_train.shape[1:], activation = 'relu'))
model.add(MaxPooling1D())
model.add(Conv1D(1000, kernel_size=5,  activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam()

model.compile(opt, loss=loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)
"""


###Pooling
"""
model = Sequential()

model.add(keras.layers.GlobalAveragePooling1D())
model.add(Dropout(0.5))
#model.add(Dense(500, activation =  keras.layers.LeakyReLU()))
#model.add(Dense(250, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam()

model.compile(opt, loss=loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

"""
#
hist = model.history

data = {}

for key in hist.history:
    data[key] = hist.history[key]

with open(f'{dir_}/{model_type}/History.pkl', 'wb') as f:
    pickle.dump(data, f)

model.load_weights(checkpoint_dir)

pred = model.predict_classes(X_test).reshape(len(X_test))

matrix = confusion_matrix(y_test, pred)
