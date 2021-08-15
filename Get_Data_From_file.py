import multiprocessing
import pandas as pd
from newspaper import Article, fulltext
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import datetime
import pickle
import time

exclude = ['AAPL', 'AMZN', 'BABA', 'BRK', 'FB', 'GOOG', 'JNJ', 'JPM', 'MSFT', 'NVDA', 'TSLA', 'TSM', 'UNH', 'V', 'WMT']

with open(f'failed_list_news.pkl', 'rb') as f:
    failed_list = pickle.load(f)
    
with open(f'complete_list_news.pkl', 'rb') as f:
    complete_list = pickle.load(f)

process_count = 4

file_loc = 'D:/Thesis_Data/News2'

df = pd.read_csv('collated_dataset.csv')
df['Done'] = ''


def get_news_text(num, href, return_dict):
    try:
        opt = webdriver.ChromeOptions()
        opt.add_argument('--headless')
        
        news_browser = webdriver.Chrome('chromedriver.exe', chrome_options=opt)
        news_browser.get(href)
        
        text = fulltext(news_browser.page_source)
        
        news_browser.close()
        
        return_dict[num] = text
    except Exception as e:
        return_dict[num] = 'Supper Error 101'
        print(f'failed {num}')
    
    if num not in return_dict:
        print('Not going in')
    

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    proc = {}
    detes={}
    i = 0
    while True:
        
        if len(proc) < process_count:
            
            if i == len(df):
                break
            if i not in complete_list:
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
                        exist = True
                
                if asset in exclude:
                    exist = True
                
                if exist == False:
                    detes[i] = {'date':date,
                                'name':name,
                                'asset':asset,
                                'Author':author
                                }
                    proc[i] = multiprocessing.Process(target = get_news_text, args=[i, href, return_dict])
                    proc[i].start()
                    time.sleep(0.5)
                    if date not in os.listdir(f"{file_loc}/{asset}"):
                        os.makedirs(f'{file_loc}/{asset}/{date}')
                else:
                    complete_list.append(i)
            print(i)
            i += 1
        
        drop = []
        
        _failed_ = False
        
        for key in proc.keys():
            _p_ = proc[key]
            if _p_.is_alive() == False:
                
                try:
                    out = return_dict[key]
                    if out != 'Supper Error 101':
                        to = detes[key]['date']
                        publisher = detes[key]['Author']
                        name = detes[key]['name']
                        today = datetime.datetime.strptime(to, '%Y-%m-%d').strftime('%Y-%m-%dT%H-%M-%S')
                        
                        file = open( f'{file_loc}'+f'/{asset}/{to}/{publisher}___{name}___{today}'.replace(',','').replace("'", " ").replace('?', '').replace('.', '_').replace(':', '').replace(';',''),'w')
                        file.write(out)
                        file.close()
                except Exception as e:
                    failed_list.append(key)
                    _failed_ = True
                
                complete_list.append(key)
                drop.append(key)
                proc[key].terminate()
        
        if _failed_ == True:
            print('Task Failed')
            with open('failed_list_news.pkl', 'wb') as f:
                pickle.dump(failed_list,f)
        
        dropped = False
        
        for dr in drop:
            dropped = True
            del proc[dr]
            del detes[dr]
            if dr in return_dict:
                del return_dict[dr]
        
        if dropped == True:
            with open('complete_list_news.pkl', 'wb') as f:
                    pickle.dump(complete_list,f)

