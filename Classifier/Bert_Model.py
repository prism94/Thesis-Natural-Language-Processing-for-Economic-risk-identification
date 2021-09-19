import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFAutoModel, AlbertTokenizerFast
import tensorflow as tf
import tensorflow_addons as tfa


#Load Data
with open('Word_Arrays.pkl', 'rb') as f:
    training, testing = pickle.load(f)

seq_len = 50
batchs = 6

model_name = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)
#tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

def data_tokenised(data, tok, seq_len):
    
    samples = len(data)
    num_labels = data[:, 1].astype(int).max()
    
    x_ids = np.zeros((samples, seq_len))
    x_mask = np.zeros((samples, seq_len))
    
    for i in range(len(training)):
        _input_ = data[i, 0]
        _tok_ =  tok.encode_plus(_input_, 
                                 max_length = seq_len, 
                                 truncation=True,
                                 padding='max_length', 
                                 add_special_tokens=True,
                                 return_tensors='tf'
                                 )
        x_ids[i, :] = _tok_['input_ids']
        x_mask[i, :] = _tok_['attention_mask']
    
    lab = np.zeros((samples, num_labels+1))
    
    lab[np.arange(samples), data[:, 1].astype(int)] = 1
    lab = lab[:, 1].reshape(-1, 1)
    
    
    return x_ids, x_mask, lab

training = data_tokenised(training, tokenizer, seq_len)
testing = data_tokenised(testing, tokenizer, seq_len)

train_dataset = tf.data.Dataset.from_tensor_slices((training))
test_dataset = tf.data.Dataset.from_tensor_slices((testing))

def bert_transform(ids, masks, lab):
    return {'input_ids':ids, 'attention_mask':masks}, lab

train_dataset = train_dataset.map(bert_transform)
test_dataset = test_dataset.map(bert_transform)

train_dataset = train_dataset.shuffle(len(training[0])).batch(batchs, drop_remainder=True)
test_dataset = test_dataset.shuffle(len(testing[0])).batch(batchs, drop_remainder=True)

###Train Model

BERT = TFAutoModel.from_pretrained(model_name)

input_ids = tf.keras.layers.Input(shape=(seq_len, ), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(seq_len, ), name='attention_mask', dtype='int32')

embeddings = BERT.bert(input_ids, attention_mask=mask)[1]
#embeddings = BERT.albert(input_ids, attention_mask=mask)[1]

den_1 = tf.keras.layers.Dense(50, activation='relu')(embeddings)
#den_2  = tf.keras.layers.Dense(1000, activation='relu')(den_1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(den_1)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)

#opt = tfa.optimizers.AdamW()
opt = tf.keras.optimizers.Adam(lr=1e-5)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(opt, loss=loss, metrics=['accuracy'])

#model.layers[1].trainable = False
#model.layers[0].trainable = False
model.layers[2].trainable = False

"""
history = model.fit(train_dataset, 
                    validation_data = test_dataset,
                    epochs = 100,
                    )
"""
