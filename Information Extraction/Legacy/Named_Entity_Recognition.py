import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.optimizers import Adam
import numpy as np

model_loc = 'D:/Thesis_Model/Information_Extraction'

with open(f'{model_loc}/Token_Data.pkl', 'rb') as f:
    X, y, tok, embedding_matrix = pickle.load(f)

max_length = 1000
split = 0.75

X_train = X[:int(len(X)*split)]
y_train = y[:int(len(y)*split)]

X_test = X[:int(len(X)*split)]
y_test = y[:int(len(y)*split)]

###Build Model
breakpoint()
model = Sequential()

model.add(Embedding(len(embedding_matrix), 300, weights=[embedding_matrix], input_length=max_length, trainable=False))

model.add(Bidirectional(LSTM(500, return_sequences=True)))

model.add(Dense(y.shape[2]), activation='sigmoid')

opt = Adam()

model.compile(loss='categorical_crossentropy', optimizer=opt)

model.fit(X_train, y_train,
          validation_data = (X_test, y_test),
          verbose = 1
         )

breakpoint()
