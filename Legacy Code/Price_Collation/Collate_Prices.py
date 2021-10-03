import pandas as pd
import os
import yfinance as yf
import time

df = pd.read_csv('C:/Users/User/OneDrive/Python_scripts/Thesis/Copy of All_Stocks_US.csv', encoding='latin-1')
price_loc = 'D:/Thesis_Data/Price'

for i in range(len(df)):
    
    tick = df['Ticker'].values[i].replace(' ','')
    print(i, tick)
    if f'{tick}.csv' not in os.listdir(price_loc):
        ticker = yf.Ticker(tick)
        data = ticker.history(period='10y')
        if len(data) > 0:
            data.to_csv(f'{price_loc}/{tick}.csv')
    time.sleep(1)


