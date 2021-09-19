from gensim.models import word2vec, KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

#with open('Word_Arrays.pkl', 'rb') as f:
    #training, testing = pickle.load(f)

#s = training[:, 0]
#s = word2vec.Text8Corpus('wordcorpus')

#model = word2vec.Word2Vec(s, vector_size=250, hs=1,)

#model.save('w2v')
loc = 'D:/Thesis_Data/Word_Embeddings/'

#glove = datapath(f'{loc}/glove.6B.300d.txt')

#tmp_file = get_tmpfile(f"{loc}/test_word2vec.txt")

#w2v = glove2word2vec(glove, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

array_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'

