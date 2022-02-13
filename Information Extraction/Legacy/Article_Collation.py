import pandas as pd
import datetime as dt
import os
import numpy as np

classified = pd.read_csv('C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier/Legacy Code/collated_dataset_Classified.csv')

output = 'D:/Thesis_Data/Lawsuits Collated'

threshold = 0.7

df = classified[classified['Prediction'] > threshold]

folders = ['News', 'News2']

file_locs = 'D:/Thesis_Data'

for i in range(len(df)):
    print(i)
    date = dt.datetime.strptime(df['Date'].values[i], '%d/%m/%Y').strftime('%Y-%m-%d')
    asset = df['Asset'].values[i]
    head = df['Headline'].values[i]
    
    for fold in folders:
        
        loc = f'{file_locs}/{fold}'
        dirs = os.listdir(loc)
        
        if asset in dirs:
            loc = f'{loc}/{asset}'
            dirs = os.listdir(loc)
            if date in dirs:
                loc = f'{loc}/{date}'
                dirs = os.listdir(loc)
                for text in dirs:
                    if head.replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';','') in text:
                        try:
                            file_1 = open(f'{loc}/{text}', 'r')
                            file_text = file_1.read()
                            file_1.close()
                            rand = np.random.randint(1, 1000)
                            file_2 = open(f'{output}/{rand}-{date}-{asset}-{text}.txt', 'w')
                            file_2.write(file_text)
                            file_2.close()
                        except:
                            print('did not work')
                
        
    
