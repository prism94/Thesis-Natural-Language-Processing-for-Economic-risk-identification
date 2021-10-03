import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import datetime
from pytz import timezone
import os
from newspaper import Article, fulltext

checker = True

WINDOW_SIZE = "1920,1080"
opt = webdriver.ChromeOptions()
opt.add_argument('--headless')
#opt.add_argument("--window-size=%s" %WINDOW_SIZE)

file_location = 'D:/Thesis_Data/News'
_timezone = timezone('US/Eastern')

finish_date = _timezone.localize(datetime.datetime(2019, 1, 1))

asset = 'UNH'

if asset not in os.listdir(file_location):
    os.makedirs(f'{file_location}/{asset}')

browser = webdriver.Chrome('chromedriver.exe')

browser.get('https://www.investing.com/')

input_box = browser.find_element_by_xpath('//input[@autocomplete="off"]').send_keys(asset)

time.sleep(10)

results = browser.find_element_by_xpath('//div[@class="js-table-results tableWrapper"]')

links = results.find_elements_by_tag_name('a')
links[0].click()
#for link in links:
    
navbar = browser.find_element_by_xpath('//div[@class="navbar_navbar-main__3rp-R"]')

navs = navbar.find_elements_by_tag_name('li')

for nav in navs:
    if nav.text == 'News & Analysis':
        nav.click()
        break

while checker == True:
    news_articles = browser.find_element_by_xpath('//div[@class="mediumTitle1"]')
    
    articles = news_articles.find_elements_by_tag_name('article')
    
    for article in articles:
        href = article.find_element_by_tag_name('a').get_attribute('href')
        name = article.find_elements_by_tag_name('a')[1].get_attribute('title')
        try:
            news_browser = webdriver.Chrome('chromedriver.exe', chrome_options=opt)
            news_browser.get(href)
            
            text = fulltext(news_browser.page_source)
            
            #art = Article(href)
            #art.download()
            #art.parse()
            #text = art.text
            
            news_browser.close()
            
            span = article.find_elements_by_tag_name('span')
            
            if len(span) < 2:
                span_text = span.text
                
                publisher = span_text.split(' - ')[0].split('By ')[1]
                _time_ = span_text.split(' - ')[1]
            else:
                publisher = span[0].text
                _time_ = span[-1].text.split(' - ')[1]
            
            done = False
            if 'hour' in _time_:
                if _time_[1] == ' ':
                    hour = _time_[0]
                else:
                    hour = _time_[:2]
                hour = int(hour)
                timer = datetime.datetime.now().astimezone(_timezone) - datetime.timedelta(hours=1)
                done = True
            if 'minute' in _time_:
                timer = datetime.datetime.now().astimezone(_timezone)
                done = True
            
            if done == False:
                timer = _timezone.localize(datetime.datetime.strptime(_time_, '%b %d, %Y'))
            
            today = timer.strftime('%Y-%m-%d-T%H-%M-%S')
            
            if timer < finish_date:
                checker = False
            
            to = timer.strftime('%Y-%m-%d')
            
            if to not in os.listdir(f'{file_location}/{asset}'):
                os.makedirs(f'{file_location}/{asset}/{to}')
            
            file = open( f'{file_location}/{asset}/{to}/{publisher}___{name}___{today}'.replace(',','').replace("'", " ").replace('?', ''),'w')
            file.write(text)
            file.close()
        except Exception as e:
            print(f'Did not work for : {name}')
            print(e)
        
    next_button = browser.find_element_by_xpath('//div[@id="paginationWrap"]')
    next_button = next_button.find_elements_by_tag_name('div')
    
    for _next_ in next_button:
        if _next_.text == 'Next':
            a = _next_.find_element_by_tag_name('a').get_attribute('href')
            browser.get(a)
    
    #next_button.click()

