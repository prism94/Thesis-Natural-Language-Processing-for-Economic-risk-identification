import pandas as pd
import numpy as np
import pickle

data_loc = 'C:/Users/User/OneDrive/Python_scripts/Thesis/Price_Collation'
file_name = 'Price_Classified_Normalised.csv'

train_sample = 25000
test_sample = 25000


df = pd.read_csv(f'{data_loc}/{file_name}')
df['Date'] = pd.to_datetime(df['Date'], format = '%Y-%m-%d')

nulls = ~df['Headline'].isnull()
df = df[nulls]

nulls = ~df['Class'].isnull()
df = df[nulls]

training_dates = [pd.to_datetime('2018-01-01', format = '%Y-%m-%d'), pd.to_datetime('2020-01-01', format = '%Y-%m-%d')]

train = df['Date'] >= training_dates[0]
train = df[train]

l = train['Date'] <= training_dates[1]
train = train[l]

l = df['Date'] > training_dates[1]
test = df[l]

train = train[['Headline', 'Class']].values
test = test[['Headline', 'Class']].values

np.random.shuffle(train)
np.random.shuffle(test)

train = train[:train_sample]
test = test[:test_sample]

package = train, test

with open('Word_Arrays.pkl', 'wb') as f:
    pickle.dump(package, f)
