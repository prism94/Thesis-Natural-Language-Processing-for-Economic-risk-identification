import pickle
import numpy as np
import pandas as pd
from gensim.models import word2vec, KeyedVectors
import tensorflow as tf

max_padding = 5
vec_length = 300
seq_length = 6011

array_loc = 'D:/Thesis_Model'

vec_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'
#with open(f'{array_loc}/Word_Arrays_Article.pkl', 'rb') as f:
    #training, testing = pickle.load(f)

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


def pipelining(X, y, mod=w2v, vec_length=vec_length, seq_length=seq_length):
    breakpoint()
    print(tf.as_string(X))
    _X_ = sentence_to_sequence(mod, X, vec_length)
    _X_ = adjust_seq_length(_X_, seq_length, vec_length)
    return _X_, int(y)

def tester(X):
    X = f'{X}_1'
    return X
