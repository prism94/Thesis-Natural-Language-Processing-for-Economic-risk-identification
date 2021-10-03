import pandas as pd
import numpy as np
import pickle
import string

file_dir = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier'

training_split = 0.75

law = pd.read_csv('USNEWS_Classified_Full.csv')
bus = pd.read_csv('USNEWS_Business_Classified_Full.csv')

law_gl = pd.read_csv('GlobalNews_Classified_Full.csv')
bus_gl =  pd.read_csv('GlobalNews_Business_Classified_Full.csv')


def drop_lawsuits(bus, law):
    bus['Overlap'] = False
    for i in range(len(bus)):
        date = bus['Date'].values[i]
        head = bus['Article'].values[i]
        
        lookup = law['Date'] == date
        law_l = law[lookup]
        
        if len(law_l) > 0:
            lookup = law_l['Article'] == head
            law_l = law_l[lookup]
            
            if len(law_l) > 0:
                bus['Overlap'].values[i] = True
    bus = bus[bus['Overlap'] == False]
    del bus['Overlap']
    return bus

bus = drop_lawsuits(bus, law)
bus_gl = drop_lawsuits(bus_gl, law_gl)

def fix_text(text):
    text=text.replace('Iâ€™', "'")
    text=text.replace('â€™', "'")
    
    tab = str.maketrans("", "", string.punctuation)
    return text.translate(tab)

def label_data(a, lab):
    df = pd.DataFrame(a)
    df['Label'] = lab
    return df.values

bus = bus[~bus['Article'].isnull()]
bus_gl = bus_gl[~bus_gl['Article'].isnull()]

law = law[~law['Article'].isnull()]
law_gl = law_gl[~law_gl['Article'].isnull()]

bus['Article'] = bus['Article'].str.lower()
bus_gl['Article'] = bus_gl['Article'].str.lower()

bus['Article'] = bus['Article'].map(fix_text)
bus_gl['Article'] = bus_gl['Article'].map(fix_text)


bus_data = bus['Article'].values
bus_data = np.append(bus_data, bus_gl['Article'].values)

law['Article'] = law['Article'].str.lower()
law_gl['Article'] = law_gl['Article'].str.lower()

law['Article'] = law['Article'].map(fix_text)
law_gl['Article'] = law['Article'].map(fix_text)

law_data = law['Article'].values
law_data = np.append(law_data,  law_gl['Article'].values)

law_data = label_data(law_data, 1)
bus_data = label_data(bus_data, 0)

np.random.shuffle(law_data)
np.random.shuffle(bus_data)

training_law = law_data[:int(len(law_data)*training_split), :]
testing_law = law_data[int(len(law_data)*training_split):, :]

training_bus = bus_data[:int(len(bus_data)*training_split), :]
testing_bus = bus_data[int(len(bus_data)*training_split):, :]


training = np.array(training_law.tolist() + training_bus.tolist())
testing = np.array(testing_law.tolist() + testing_bus.tolist())

np.random.shuffle(training)
np.random.shuffle(testing)

package = training, testing

with open(f'Word_Arrays_Article.pkl', 'wb') as f:
    pickle.dump(package, f)
