import pickle
import numpy as np
import copy

dir_ = 'D:/Thesis_Model/Full Articles_2'
model_loc = 'D:/Thesis_Model'

cutoff = 10

problematic_words = ['139']

with open(f'{model_loc}/Word_Arrays_Article.pkl', 'rb') as f:
    training, testing = pickle.load(f)

dataset = np.vstack((training, testing))

def get_phrases(word, data, cutoff):
    phrases = []
    for i in range(len(data)):
        if word in data[i,0]:
            splits = data[i,0].split('    ')
            for split in splits:
                if word in split:
                    phrases.append(split)
    
    phrase_counter = {}
    
    for phrase in phrases:
        if phrase not in phrase_counter:
            phrase_counter[phrase] = 1
        else:
            phrase_counter[phrase] += 1
    
    del_key = []
    
    for key in phrase_counter.keys():
        if phrase_counter[key] < cutoff:
            del_key.append(key)
    
    for key in del_key:
        del phrase_counter[key]
    
    return phrase_counter

problems = {}

for prob in problematic_words:
    phrases = get_phrases(prob ,dataset, cutoff)
    problems[prob] = phrases
    
###Fix dataset

def remove_paragraphs(data, problems):
    df = copy.copy(data)
    
    phrases = []
    for prob in problems.keys():
        for p in problems[prob].keys():
            phrases.append(p)
    
    for i in range(len(data)):
        text = data[i, 0]
        cont = False
        for p in phrases:
            if p in text:
                cont = True
        if cont == True:
            splits = text.split('    ')
            new_text = ''
            for s in splits:
                if s not in phrases:
                    new_text = f'{new_text}    {s}'
            df[i, 0]=new_text
    
    return df


training = remove_paragraphs(training, problems)
testing = remove_paragraphs(testing, problems)

package = training, testing

#with open(f'{model_loc}/Word_Arrays_Article_Altered.pkl', 'wb') as f:
    #pickle.dump(package, f)
