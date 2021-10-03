import pickle
import numpy as np
import pandas as pd
from gensim.models import word2vec, KeyedVectors

max_padding = 5
vec_length = 300

array_loc = 'D:/Thesis_Model'
vec_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'

with open(f'{array_loc}/Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

#w2v = word2vec.Word2Vec.load('w2v') if for pretrained
w2v = KeyedVectors.load(f'{vec_loc}/glove.modeel')


def sentence_to_sequence(model, text, vec_length):
    splits = text.split(' ')
    sequence = []
    
    for s in splits:
        if s in model.key_to_index:
            sequence.append(model.get_vector(s))
            #sequence.append(model.wv[s].tolist()) #If not glove
        else:
            sequence.append([0 for i in range(vec_length)])
    
    return sequence

def adjust_seq_length(seq, seq_length, vec_length):
    _len_ = len(seq)
    to_create = seq_length - _len_
    if to_create < 0:
        return np.NaN
    else:
        empties = [[0 for i in range(vec_length)] for n in range(to_create)]
        return empties + seq


training = pd.DataFrame(training)
testing = pd.DataFrame(testing)

training['wv'] = training[0].apply(lambda row: sentence_to_sequence(w2v, row, vec_length))
testing['wv'] = testing[0].apply(lambda row: sentence_to_sequence(w2v, row, vec_length))

training['length'] = training['wv'].apply(lambda x: len(x))

max_length = training['length'].max()
word_sequence_length = max_length + max_padding

training['input'] = training['wv'].apply(lambda x: adjust_seq_length(x, word_sequence_length, vec_length))
testing['input'] = testing['wv'].apply(lambda x: adjust_seq_length(x, word_sequence_length, vec_length))

testing = testing[~testing['input'].isnull()]

training[1] = training[1].astype(int)
testing[1] = testing[1].astype(int)

y_train = training[1].values

X_train = training['input'].values.tolist()
X_train = np.array(X_train)

y_test = testing[1].values

X_test = testing['input'].values.tolist()
X_test = np.array(X_test)

package = X_train, y_train, X_test, y_test

with open('Modeling_Data.pkl', 'wb') as f:
    pickle.dump(package, f)
