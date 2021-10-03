from gensim.models import word2vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import pickle
import numpy as np


max_length = 6000
num_w = 85000
vec_length = 300

vec_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'

w2v = KeyedVectors.load(f'{vec_loc}/glove.modeel')


array_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'
model_loc = 'D:/Thesis_Model'

with open(f'{array_loc}/Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

#Tokens
X_train = training[:, 0]
y_train = training[:, 1]

X_test = testing[:, 0]
y_test = testing[:, 1]

tok = Tokenizer(num_words=num_w)
tok.fit_on_texts(X_train)

train = tok.texts_to_sequences(X_train)
test = tok.texts_to_sequences(X_test)

X_train = pad_sequences(train, padding='post', truncating = 'post', maxlen=max_length)
X_test = pad_sequences(test, padding='post', truncating = 'post', maxlen=max_length)

vocab_size = len(tok.word_index) + 1

embeddings_index = dict()
f = open('D:/Thesis_Data/Word_Embeddings/glove.6B.300d.txt', 'r', encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()


# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tok.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

package = tok, embedding_matrix

with open(f'{model_loc}/Token_Data.pkl', 'wb') as f:
    pickle.dump(package, f)

"""
def embedding_matrix_creation(w2v, tok):
    matrix = []
    
    token_index = tok.index_word
    words = w2v.key_to_index
    for key in token_index.keys():
        word = token_index[key]
        if word in words:
            matrix.append(w2v.word_vec(word).tolist())
        else:
            matrix.append([0 for _ in range(vec_length)])
    #matrix.append([0 for _ in range(vec_length)])
    return np.array(matrix)

#embedding_matrix = embedding_matrix_creation(w2v, tok)

#package = tok, embedding_matrix

#with open(f'{model_loc}/Token_Data.pkl', 'wb') as f:
    #pickle.dump(package, f)

"""
