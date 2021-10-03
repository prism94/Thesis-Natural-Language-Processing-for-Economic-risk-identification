import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import pickle
import time

df = pd.read_csv('USNEWS.csv')

options = webdriver.ChromeOptions() 
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome('chromedriver.exe', options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})

data = {'Index':[],
        'Tags':[]}

with open(f'USNEWS_Tags.pkl', 'rb') as f:
    data = pickle.load(f)

for i in range(len(df)):
    if i not in data['Index']:
        href = df['Href'].values[i]
        driver.get(href)
        
        code = bs(driver.page_source)
        
        tags = code.find_all('div', {'class':'Box-w0dun1-0 egjbLe'})
        
        tagers = []
        
        if len(tags) > 0:
            a_s = tags[0].find_all('a')
            for a in a_s:
                text = a.text
                tagers.append(text)
        
        data['Index'].append(i)
        data['Tags'].append(tagers)
        
        with open(f'USNEWS_Tags.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        time.sleep(1)
