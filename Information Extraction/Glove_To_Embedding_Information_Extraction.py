from gensim.models import word2vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import copy

vec_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier/Legacy Code'

w2v = KeyedVectors.load(f'{vec_loc}/glove.modeel')


array_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Information_Extraction'
model_loc = 'D:/Thesis_Model/Information_Extraction'

with open(f'{array_loc}/Compiled_Labeled_Data_Final.pkl', 'rb') as f:
    training_data = pickle.load(f)

max_length = 1000
num_w = 85000
vec_length = 300

max_len = 512

split = 0.75

tok = Tokenizer(num_words=num_w)
texts = pd.DataFrame(training_data)[0].values
train_texts = texts[:int(len(texts)*split)]
tok.fit_on_texts(train_texts)


def labeling_to_output(train):

  text = train[0]
  tokens = train[1]
  stringer = tokenizer.tokenize(text)
  tok = tokenizer(
                  text,
                  max_length = max_len, #change this for padding
                  truncation=True,
                  padding='max_length',
                  add_special_tokens=True,
                  return_tensors='np'

  )

  output = np.zeros((len(tok['attention_mask'][0]), len(entity_diction)))

  for tag in tokens['entities']:
    start = tag[0]
    end = tag[1]
    tager = tag[2]

    before = tokenizer(text[:start], return_tensors='np')
    after = tokenizer(text[start:+end], return_tensors='np')

    starter = len(before['input_ids'][0]) - 1
    afterer = len(after['input_ids'][0]) - 1

    output[starter:(starter+afterer-1), entity_diction[tager]] = 1

  return tok, output

def labeling_to_output_custom(train):

  text = train[0]
  tokens = train[1]
  #stringer = tok.tokenize(text)

  seq = np.array(tok.texts_to_sequences([text])[0])
  seq = np.array(pad_sequences([seq], padding='post', truncating = 'post', maxlen=max_length))[0]

  output = np.zeros((len(seq), len(entity_diction)))

  for tag in tokens['entities']:
    start = tag[0]
    end = tag[1]
    tager = tag[2]

    before = tok.texts_to_sequences([text[:start]])
    after = tok.texts_to_sequences([text[start:end]])

    starter = len(before[0])
    afterer = len(after[0])

    output[starter:(starter+afterer), entity_diction[tager]] = 1

  return seq, output

def get_entities(text, out, prob):
  output = copy.copy(out)
  output[output> prob] = 1
  output[output<= prob] = 0

  identified = []

  entity_loc = {
    2:["Reason", False, []],
    1:["Defendant",False, []],
    3:["Penalties", False, []],
    4:["Outcome", False, []],
    0:["Platiff", False, []]
  }

  for i in range(0, len(output)):
      word = text[i]
      for n in range(5):
          open = entity_loc[n][1]
          val = output[i, n]
          if open == True:
              if val == 1:
                  entity_loc[n][2].append(word)
              else:
                  entity_loc[n][1] = False
                  identified.append([entity_loc[n][0], tok.sequences_to_texts([entity_loc[n][2]])])
                  entity_loc[n][2] = []
          else:
              if val == 1:
                  entity_loc[n][2].append(word)
                  entity_loc[n][1] = True

  return identified

entity_diction = { "Reason":2,
  "Defendant":1,
  "Penalties":3,
  "Outcome":4,
  "Platiff":0
}

text = []

X = []

y = []

for i in range(len(training_data)):
    print(i)
    example = training_data[i]
    text.append(example[0])

    _X_, output = labeling_to_output_custom(example)

    X.append(_X_.tolist())
    y.append(output.tolist())



split = 0.75

X = np.array(X)
y = np.array(y)

#Tokens
"""
X_train = X[:int(len(X)*split)]
y_train = y[:int(len(y)*split)]

X_test = X[:int(len(X)*split)]
y_test = y[:int(len(y)*split)]
"""

vocab_size = len(tok.word_index) + 1

embeddings_index = dict()

f = open('D:/Thesis_Model/Word_Embeddings/glove.6B.300d.txt', 'r', encoding="utf8")

for l in f:
	values = l.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tok.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

package = X, y, tok, embedding_matrix

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
