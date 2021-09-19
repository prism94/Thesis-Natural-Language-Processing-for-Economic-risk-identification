import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif as mi

#data_dir = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'
data_dir = 'D:/Thesis_Model'
token_loc = 'D:/Thesis_Model'

with open(f'{data_dir}/Word_Arrays_Article_Altered.pkl', 'rb') as f:
    training, testing = pickle.load(f)

with open(f'{token_loc}/Token_Data_Article.pkl', 'rb') as f:
    tok, embedding_matrix = pickle.load(f)

dataset = np.vstack((training, testing))

def label_word(data, word):
    labels = []
    for i in range(len(data)):
        if word in data[i, 0].split():
            label = 1
        else:
            label = 0
        labels.append(label)
    return np.array(labels)

#test = label_word(dataset, 'lawsuit')

def correlate_metrics(word_label, label):
    
    df = pd.DataFrame({'Word_Label':word_label, 
                       'Label':label
                       })
    corr = df['Word_Label'].corr(df['Label'])
    
    mutual_information = mi(df['Word_Label'].values.reshape(-1,1), df['Label'].values)
    
    return corr, mutual_information[0]


#corr, mute = correlate_metrics(test, dataset[:, 1].astype(int))

data = {'Word':[],
        'Corr':[],
        'Mutual Information':[],
        'Count':[]
        }
count = 0
length = len(tok.word_index)
for token in tok.word_index.keys():
    print(f'{count} of {length}: {token}')
    word_labels = label_word(dataset, token)
    corr, mute = correlate_metrics(word_labels, dataset[:, 1].astype(int))
    data['Word'].append(token)
    data['Corr'].append(corr)
    data['Mutual Information'].append(mute)
    data['Count'].append(word_labels.sum())
    count += 1

df = pd.DataFrame(data)
df.to_csv('Article_Word_Analysis_Altered.csv', index=False)
