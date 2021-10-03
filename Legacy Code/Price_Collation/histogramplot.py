import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

with open('Output_distribution.pkl', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

increment = 0.02

df_l = df[df['Class'] == 1]['Result'].values
len_l = len(df_l)
df_n = df[df['Class']==0]['Result'].values

print(f'Normal Mean: {df_n.mean()} Normal STD: {df_n.std()}')
print(f'Lawsuit Mean: {df_l.mean()} Lawsuit STD: {df_l.std()}')

np.random.shuffle(df_n)
df_n = df_n[:len_l]

bin_list = []
binner = -1
while binner < 1:
    bin_list.append(binner)
    binner+=increment

plt.figure(figsize=(10,7), dpi= 80)
plt.hist(df_l, bins=bin_list, color = 'red', label='x')#, fc=(0, 0, 0.5, 0.5))
plt.hist(df_n, bins=bin_list, color='green', label='y', fc=(0, 0, 1, 0.5))
plt.legend(loc='upper right')
plt.show()

"""
plt.scatter(df_l, [1 for _ in range(len(df_l))])
plt.scatter(df_n, [0 for _ in range(len(df_n))])
plt.show()

"""
