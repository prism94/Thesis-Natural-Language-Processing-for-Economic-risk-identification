import pickle
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU, RNN, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

dir_ = 'D:/Thesis_Model'

max_length = 50
num_w = 10000

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

###

model_type = 'GlobalPooling-bag-of-words-P3'

params = 'adam-relu-dropout'

model_type = model_type + '-' + params

if model_type not in os.listdir(dir_):
    os.makedirs(f'{dir_}/{model_type}') 

#Checkpoint
checkpoint_dir = f'D:/Thesis_Model/{model_type}/Checkpoint'

checkpoint=ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

callbacks = [checkpoint]



#LSTM

model = Sequential()

model.add(Embedding(num_w, 24, input_length=max_length))
#model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.GlobalMaxPooling1D())
#model.add(Dropout(0.5))
#model.add(Dense(25, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(250, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)



"""

#RNN

model = Sequential()

model.add(SimpleRNN(1000, input_shape=X_train.shape[1:], activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(250, activation = 'relu'))
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

"""
#Transfer Learning
pre_trained = 'D:/Thesis_Model/GlobalPooling-bag-of-words-P3-adam-relu-dropout/Checkpoint'

mod = Sequential()

mod.add(Embedding(num_w, 50, input_length=max_length))
mod.add(keras.layers.GlobalAveragePooling1D())
#mod.add(Dropout(0.5))
#mod.add(Dense(500, activation = 'relu'))
#mod.add(Dropout(0.5))
#mod.add(Dense(250, activation = 'relu'))
mod.add(Dense(1, activation = 'sigmoid'))

mod.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

mod.load_weights(pre_trained)

model = Sequential()

model.add(mod.layers[0])
model.add(GRU(250))#, activation = 'relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dense(250, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

loss = keras.losses.BinaryCrossentropy()
opt = keras.optimizers.Adam()

model.compile(opt, loss=loss, metrics=['accuracy'])

model.layers[0].trainable = False

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)
"""

#Results
hist = model.history

data = {}

for key in hist.history:
    data[key] = hist.history[key]

with open(f'{dir_}/{model_type}/History.pkl', 'wb') as f:
    pickle.dump(data, f)

model.load_weights(checkpoint_dir)

pred = model.predict_classes(X_test).reshape(len(X_test))

matrix = confusion_matrix(y_test, pred)

with open(f'{dir_}/{model_type}/Token.pkl', 'wb') as f:
    pickle.dump(tok, f)
