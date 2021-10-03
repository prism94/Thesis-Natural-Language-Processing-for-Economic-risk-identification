import pandas as pd
import pickle
import os
import datetime

file_loc = 'D:/Thesis_Data/News Headlines Investing_com'

asset_loc = os.listdir(f'{file_loc}')

data = {'Date':[],
        'Asset':[],
        'Headline':[],
        'Author':[],
        'Href':[]}


for loc in asset_loc:
    print(loc)
    _files_ = os.listdir(f'{file_loc}/{loc}')
    for file in _files_:
        with open(f'{file_loc}/{loc}/{file}', 'rb') as f:
            pick = pickle.load(f)
        
        for _key_ in pick.keys():
            day = pick[_key_]
            for headline in day.keys():
                author = day[headline]['Author']
                href = day[headline]['href']
                
                data['Date'].append(_key_)
                data['Asset'].append(loc)
                data['Headline'].append(headline)
                data['Author'].append(author)
                data['Href'].append(href)
    
    
df = pd.DataFrame(data)
