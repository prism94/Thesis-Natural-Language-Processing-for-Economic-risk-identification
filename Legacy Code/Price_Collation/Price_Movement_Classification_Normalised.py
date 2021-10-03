import pandas as pd
import numpy as np
import os

df = pd.read_csv('C:/Users/User/OneDrive/Python_scripts/Thesis/collated_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Class'] = np.NaN

price_loc = 'D:/Thesis_Data/Price'
norm_asset = 'SPY'

time_frame = pd.Timedelta(days=2)

last_ass = ''

norm_p = pd.read_csv(f'{price_loc}/{norm_asset}.csv')
norm_p['Date'] = pd.to_datetime( norm_p['Date'], format='%Y-%m-%d')

for i in range(len(df)):
    
    ass = df['Asset'].values[i]
    date = df['Date'].values[i]
    
    print(ass, i)
    
    if ass != last_ass:
        _p_ = pd.read_csv(f'{price_loc}/{ass}.csv')
        _p_['Date'] = pd.to_datetime(_p_['Date'])
    
    lookup = _p_['Date'] > date
    _p_l = _p_[lookup]
    
    lookup = _p_l['Date'] <= (date + time_frame)
    _p_l = _p_l[lookup]
    
    if len(_p_l) > 0:
        
        p_look = norm_p['Date'] > date
        _n_ = norm_p[p_look]
        
        p_look = _n_['Date'] <= (date + time_frame)
        _n_ = _n_[p_look]
        
        n_start = _n_['Open'].values[0]
        n_close = _n_['Close'].values[0]
        
        n_change = (n_close-n_start)/n_start
        
        _open_ = _p_l['Open'].values[0]
        close = _p_l['Close'].values[-1]
        diff = (close-_open_)/_open_
        
        diff = diff-n_change
        
        if diff >= 0:
            class_ = 1
        else:
            class_ = 0
        
        df['Class'].values[i] = class_
    

df.to_csv('Price_Classified_Normalised.csv', index=False)
