import pandas as pd
import datetime as dt
import pickle

df = pd.read_csv('C:/Users/User/OneDrive/Python_scripts/Thesis/Classifier/collated_dataset_Classified.csv')

timeframe = pd.Timedelta(days=30)
lag_time = pd.Timedelta(days=7)

normalised_ticker = 'SPY'

threshold = 0.9

start_date = dt.datetime(2017, 12, 1)
end_date = dt.datetime(2021, 6, 1)

def classify(value, threshold):
    if value > threshold:
        classification = 1
    else:
        classification = 0
    return classification

df['Class'] = df['Prediction'].map(lambda x: classify(x, threshold))
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

price_loc = 'D:/Thesis_Data/Price'

asset_list = df['Asset'].unique()

data = {'Date':[],
        'Asset':[],
        'Class':[],
        'Result':[]}

norm_p = pd.read_csv(f'{price_loc}/{normalised_ticker}.csv')
norm_p['Date'] = pd.to_datetime(norm_p['Date'])

for ass in asset_list:
    _df_ = df[df['Asset'] == ass]
    print(ass, len(data['Date']))
    price = pd.read_csv(f'{price_loc}/{ass}.csv')
    price['Date'] = pd.to_datetime(price['Date'], format='%Y-%m-%d')
    
    price = price[price['Date'] >= start_date]
    price = price[price['Date'] <= end_date]
    
    if len(price)>0:
        last = price['Date'].values[-1]
        
        for i in range(len(price)):
            date = price['Date'].values[i]
            
            look = _df_['Date'] == date
            _df_l = _df_[look]
            
            if len(_df_l) > 0:
                prediction = _df_l['Class'].values.sum()
            else:
                prediction = 0
            
            if prediction == 0:
                before = date - lag_time
                after = date + timeframe
                if before > start_date:
                    if after < end_date:
                        if after < last:
                            look = _df_['Date'] > before
                            _df_l = _df_[look]
                            
                            look = _df_l['Date'] <= date
                            _df_l = _df_l[look]
                            
                            class_ = _df_l['Class'].values.sum()
                            
                            if class_ == 0:
                                look = _df_['Date'] > date
                                _df_l = _df_[look]
                                
                                look = _df_['Date'] < after
                                _df_l = _df_l[look]
                                
                                class_ = _df_l['Class'].values.sum()
                                
                                if class_ == 0:
                                    
                                    p_look = norm_p['Date'] > date
                                    _n_ = norm_p[p_look]
                                    
                                    p_look = _n_['Date'] < after
                                    _n_ = _n_[p_look]
                                    
                                    n_start = _n_['Open'].values[0]
                                    n_close = _n_['Close'].values[0]
                                    
                                    n_change = (n_close-n_start)/n_start
                                    
                                    look = price['Date'] > date
                                    _p_ = price[look]
                                    
                                    look = _p_['Date'] < after
                                    _p_ = _p_[look]
                                    
                                    
                                    if len(_p_) > 0:
                                        start = _p_['Open'].values[0]
                                        close = _p_['Close'].values[-1]
                                        change = (close-start)/start
                                        
                                        net_change = change-n_change
                                        
                                        data['Date'].append(date)
                                        data['Asset'].append(ass)
                                        data['Class'].append(0)
                                        data['Result'].append(net_change)
            else:
                before = date - lag_time
                after = date + timeframe
                if before > start_date:
                    if after < end_date:
                        if after < last:
                            look = _df_['Date'] > before
                            _df_l = _df_[look]
                            
                            look = _df_l['Date'] < date
                            _df_l = _df_l[look]
                            
                            class_ = _df_l['Class'].values.sum()
                            
                            if class_ == 0:
                                look = _df_['Date'] > date
                                _df_l = _df_[look]
                                
                                look = _df_['Date'] < after
                                _df_l = _df_l[look]
                                
                                class_ = _df_l['Class'].values.sum()
                                
                                if class_ == 0:
                                    
                                    p_look = norm_p['Date'] > date
                                    _n_ = norm_p[p_look]
                                    
                                    p_look = _n_['Date'] < after
                                    _n_ = _n_[p_look]
                                    
                                    n_start = _n_['Open'].values[0]
                                    n_close = _n_['Close'].values[0]
                                    
                                    n_change = (n_close-n_start)/n_start
                                    
                                    look = price['Date'] > date
                                    _p_ = price[look]
                                    
                                    look = _p_['Date'] < after
                                    _p_ = _p_[look]
                                    
                                    if len(_p_) > 0:
                                       start = _p_['Open'].values[0]
                                       close = _p_['Close'].values[-1]
                                       change = (close-start)/start
                                       
                                       net_change = change-n_change
                                       
                                       data['Date'].append(date)
                                       data['Asset'].append(ass)
                                       data['Class'].append(1)
                                       data['Result'].append(net_change)
                
    
with open('Output_distribution.pkl', 'wb') as f:
    pickle.dump(data, f)
