from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import pandas as pd
import datetime
import time
import pickle
"""
press = 100

options = webdriver.ChromeOptions() 
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome('chromedriver.exe', options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})

driver.get('https://globalnews.ca/tag/lawsuit/ ')
"""

test = open('global_website_business.txt', 'r', encoding="utf8").read()

code = bs(test, 'lxml')

#stories = code.find('ul', {'id':'archive-latestStories'})

lis = code.find_all('li', {'class':'c-posts__item c-posts__loadmore'})

data = {'Date':[],
        'Asset':[],
        'Headline':[],
        'Author':[],
        'Href':[]}

for li in lis:
    try:
        headline = li.find('span', {'class':'c-posts__headlineText'}).text
        asset = 'globalnews'
        date = li.find('div', {'class':'c-posts__about c-posts__about--multiple'}).find_all('div')[1].text
        if ',' in date:
            format_ = '%b %d, %Y'
        else:
            format_ = '%b %d'
        date = datetime.datetime.strptime(date, format_)
        date = date.strftime('%Y-%m-%d')
        author = li.find('div', {'class':'c-posts__about c-posts__about--multiple'}).find_all('div')[0].text
        href = li.find('a').get_attribute_list('href')[0]
        
        data['Date'].append(date)
        data['Asset'].append(asset)
        data['Headline'].append(headline)
        data['Author'].append(author)
        data['Href'].append(href)
    except Exception as e:
        print(e)

df = pd.DataFrame(data)

df.to_csv('GlobalNews_Business.csv', index=False)
