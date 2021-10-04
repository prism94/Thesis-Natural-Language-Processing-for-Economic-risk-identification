import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import datetime
from pytz import timezone
import os
from newspaper import Article, fulltext
import pickle


def fetch_data(asset, finish_date):
    checker = True
    try:
        WINDOW_SIZE = "1920,1080"
        opt = webdriver.ChromeOptions()
        opt.add_argument('--headless')
        #opt.add_argument("--window-size=%s" %WINDOW_SIZE)
        
        file_location = 'D:/Thesis_Data/News Headlines Investing_com'
        _timezone = timezone('US/Eastern')
        
        #finish_date = _timezone.localize(datetime.datetime(2018, 1, 1))
        
        if asset not in os.listdir(file_location):
            os.makedirs(f'{file_location}/{asset}')
        
        browser = webdriver.Chrome('chromedriver.exe', chrome_options=opt)
        
        browser.get('https://www.investing.com/')
        
        input_box = browser.find_element_by_xpath('//input[@autocomplete="off"]').send_keys(asset)
        
        time.sleep(10)
        
        results = browser.find_element_by_xpath('//div[@class="js-table-results tableWrapper"]')
        
        links = results.find_elements_by_tag_name('a')
        
        for l in links:
            t = l.text.split(' ')[0]
            end = l.text.split('- ')[-1]
            if t == asset:
                if end == 'NYSE' or end == 'NASDAQ':
                    l.click()
                    break
        
        time.sleep(5)
        
        #for link in links:
            
        navbar = browser.find_element_by_xpath('//div[@class="navbar_navbar-main__3rp-R"]')
        
        navs = navbar.find_elements_by_tag_name('li')
        
        for nav in navs:
            if nav.text == 'News & Analysis':
                nav.click()
                break
        
        data = {}
        last_month = None
        last_year = None
        
        while checker == True:
            news_articles = browser.find_element_by_xpath('//div[@class="mediumTitle1"]')
            
            articles = news_articles.find_elements_by_tag_name('article')
            
            for article in articles:
                
                href = article.find_element_by_tag_name('a').get_attribute('href')
                name = article.find_elements_by_tag_name('a')[1].get_attribute('title')
                
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
                
                timer = datetime.datetime(timer.year, timer.month, timer.day)
                
                if timer < finish_date:
                    checker = False
                
                if last_month == None:
                    last_year = timer.year
                    last_month = timer.month
                
                if last_month != timer.month:
                    with open(f'{file_location}/{asset}/{last_year}-{last_month}', 'wb') as f:
                        pickle.dump(data, f)
                    data = {}
                    last_year = timer.year
                    last_month = timer.month
                
                if timer not in data:
                    data[timer] = {}
                
                data[timer][name] = {'Author':publisher,
                                    'href':href
                                    }
                        
            next_button = browser.find_element_by_xpath('//div[@id="paginationWrap"]')
            next_button = next_button.find_elements_by_tag_name('div')
            
            for _next_ in next_button:
                if _next_.text == 'Next':
                    a = _next_.find_element_by_tag_name('a').get_attribute('href')
                    browser.get(a)
            
            #next_button.click()
        browser.close()
    except Exception as e:
        print(f'Error: {asset}')
        with open('Error_List.pkl', 'rb') as f:
            errors = pickle.load(f)
        
        errors.append(asset)
    
        with open('Error_List.pkl', 'wb') as f:
            pickle.dump(errors, f)

