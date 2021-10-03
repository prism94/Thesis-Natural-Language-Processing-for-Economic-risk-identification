import os
import pandas as pd
import datetime

words = ['Lawsuit', 'Litigation', 'countersuit', 'counterclaim']

file_loc = 'D:/Thesis_Data/News2'

df = pd.read_csv('collated_dataset.csv')
df['ShortList'] = ''

for i in range(len(df)):
    print(i)
    date = df['Date'].values[i]
    name = str(df['Headline'].values[i])
    asset = df['Asset'].values[i]
    href = df['Href'].values[i]
    author = df['Author'].values[i]
    
    concat = f'{author}___{name}___{datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%dT%H-%M-%S")}'.replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';','')
    
    exist = False
    if asset in os.listdir(f'{file_loc}'):
        if date in os.listdir(f'{file_loc}/{asset}'):
            if concat in os.listdir(f'{file_loc}/{asset}/{date}'):
                exist = True
    
    if exist == True:
        
        file = open(f'{file_loc}/{asset}/{date}/{concat}', 'r')
        text = file.read()
        text = text.replace('\n', ' ')
        text = text.lower()
        wording = False
        for w in words:
            if w.lower() in text:
                wording = True
        
        if wording == True:
            df['ShortList'].values[i] = 'Yes'

df.to_csv('shortlisted.csv')
