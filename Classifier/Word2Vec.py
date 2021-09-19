from gensim.models import word2vec
import pickle

with open('Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

#s = training[:, 0]
s = word2vec.Text8Corpus('wordcorpus')

model = word2vec.Word2Vec(s, vector_size=250, hs=1,)

model.save('w2v')
