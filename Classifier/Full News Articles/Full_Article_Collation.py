import pandas as pd
import os
import datetime

file_loc = 'D:/Thesis_Data/News_Lawsuits'

file_name = 'USNEWS_Classified'

file = f'C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier/{file_name}.csv'

df = pd.read_csv(file)
def format_date(data):
    data = pd.to_datetime(data, format='%d/%m/%Y')
    data = data.strftime('%Y-%m-%d')
    return data
df['Date'] = df['Date'].map(lambda x: format_date(x))

df['Article'] = ''

for i in range(len(df)):
    date = df['Date'].values[i]
    name = str(df['Headline'].values[i]).replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';','')
    asset = df['Asset'].values[i]
    href = df['Href'].values[i]
    author = df['Author'].values[i]
    
    concat = f'{author}___{name}___{datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%dT%H-%M-%S")}'.replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';','')
    
    exist = False
    if asset not in os.listdir(file_loc):
        os.makedirs(f'{file_loc}/{asset}')
    
    if date in os.listdir(f'{file_loc}/{asset}'):
        if concat in os.listdir(f'{file_loc}/{asset}/{date}'):
            if date in os.listdir(f"{file_loc}/{asset}"):
                exist = True
    
    if exist == True:
        today = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%dT%H-%M-%S')
        file = open( f'{file_loc}'+f'/{asset}/{date}/{author}___{name}___{today}'.replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';',''),'r')
        text = file.read().replace('\n', '  ')
        file.close()
        df['Article'].values[i] = text
    
    if exist == False:
        print(i)
    
df.to_csv(f'{file_name}_Full.csv', index=False)
