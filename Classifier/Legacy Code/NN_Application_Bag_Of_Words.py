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

loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis'
file_name = 'collated_dataset'

df = pd.read_csv(f'{loc}/{file_name}.csv')

dir_ = 'D:/Thesis_Model'
model_type = 'GlobalPooling-bag-of-words-P3-adam-relu-dropout'
model_dir = 'D:/Thesis_Model/GlobalPooling-bag-of-words-P3-adam-relu-dropout/Checkpoint'
max_length = 50

with open(f'{dir_}/{model_type}/Token.pkl', 'rb') as f:
    tok = pickle.load(f)

#

model = Sequential()

model.add(Embedding(10000, 24, input_length=max_length))
#model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.GlobalMaxPooling1D())
#model.add(Dropout(0.5))
#model.add(Dense(25, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(250, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)



model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

model.load_weights(model_dir)

def apply_prediction(df, model, tok):
    df = df[~df['Headline'].isnull()]
    X = pad_sequences(tok.texts_to_sequences(df['Headline'].values), padding='post', truncating='post', maxlen=max_length)
    pred = model.predict(X)
    df['Prediction'] = pred
    return df

classified = apply_prediction(df, model, tok)
classified.to_csv(f'{file_name}_Classified.csv', index=False)
