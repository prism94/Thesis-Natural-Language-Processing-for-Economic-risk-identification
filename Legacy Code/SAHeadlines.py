import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import datetime
from pytz import timezone
import os
import pickle

asset = 'TSLA'

file_loc = 'C:/Users/joshh/OneDrive/Documents/Thesis/News Headlines'


date_start = datetime.datetime(2021, 4, 1)
date_end = datetime.datetime(2019, 1, 1)

def click_dates(browser, day):
    time.sleep(3)
    
    year_list = browser.find_element_by_xpath('//div[@data-test-id="years-list"]')
    ul = year_list.find_element_by_tag_name('ul')
    
    li = ul.find_elements_by_tag_name('li')
    
    date_start_year = date_start.year
    
    for l in li:
        if str(date_start_year) == l.text:
            but = l.find_element_by_tag_name('button')
            but.click()
            break
    
    time.sleep(3)
    
    month_list = browser.find_element_by_xpath('//div[@data-test-id="months-list"]')
    
    ul = year_list.find_element_by_tag_name('ul')
    li = ul.find_elements_by_tag_name('li')
    
    date_start_month = date_start.strftime('%b')
    
    for l in li:
        if str(date_start_month) == l.text:
            but = l.find_element_by_tag_name('button')
            but.click()
            break
    
    
    
    row_list = browser.find_elements_by_xpath('//div[@role="row"]')
    laster = None
    for row in row_list:
        cols = row.find_elements_by_tag_name('div')
        for col in cols:
            
            if day != 'last':
                if col.text == day:
                    col.click()
            else:
                if col.text != '':
                    laster = col
    
    if laster != None:
        laster.click()

def scrolling(driver):
    SCROLL_PAUSE_TIME = 5
    last_height = ''
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
    
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
def collect_data(browser, dater):
    data = {}
    art_list = browser.find_element_by_xpath('//div[@data-test-id="post-list"]')
    arts = art_list.find_elements_by_tag_name('article')
    
    for art in arts:
        href = art.find_element_by_tag_name('a').get_attribute('href')
        h3 = art.find_element_by_tag_name('h3')
        head_line = h3.text
        footer = art.find_element_by_tag_name('footer')
        span = footer.find_elements_by_tag_name('span')
        author = span[0].text
        date = span[1].text
        if len(date.split('.')) > 1:
            _day_ = date.split('.')[1].replace(' ','')
        else:
            _day_ = date.split(' ')[-1].replace(' ','')
            
        if ',' in _day_:
            _day_ = _day_.split(',')[0]
       
        if int(_day_) > 31:
            _day_ = date.split(',')[1].split(' ')[-1]
       
        date = datetime.datetime(dater.year, dater.month, int(_day_))
        
        if date not in data:
            data[date] = {}
        
        data[date][head_line] = {'Author':author,
                                 'href':href
                                 }
    
    return data
        

if asset not in os.listdir(file_loc):
    os.makedirs(f'{file_loc}/{asset}')

browser = webdriver.Chrome('chromedriver.exe')

browser.get(f'https://seekingalpha.com/symbol/{asset}')

time.sleep(10)
    
el = browser.find_elements_by_xpath('//a[@data-test-id="card-footer-link"]')

for e in el:
    if e.text == 'See All News »':
        e.click()
        break

time.sleep(5)

# Get rid of pop up

time.sleep(3)

while date_start > date_end:
    date_start = date_start - datetime.timedelta(days=1)
    try:
        grids = browser.find_elements_by_xpath('//div[@data-test-id="grid-col"]')
        
        for g in grids:
            if 'Date Range:' in g.text:
                test = g
        
        button = test.find_element_by_tag_name('button')
        button.click()
        print('clicked dates')
        """
        find_date_button = browser.find_elements_by_xpath('//button[@data-test-id="dropdown"]')
        
        for button in find_date_button:
            if button.text == 'Select Date':
                button.click()
            """
        time.sleep(3)
        
        date_picker_from = browser.find_element_by_xpath('//button[@data-test-id="date-picker-from"]')
        date_picker_from.click()
        
        #Click
        click_dates(browser, '1')
        
        print('clicked first')
        
        time.sleep(3)
        
        date_picker_from = browser.find_element_by_xpath('//button[@data-test-id="date-picker-to"]')
        date_picker_from.click()
        
        time.sleep(3)
        
        #Click
        click_dates(browser, 'last')
        print('clicked last')
        
        time.sleep(3)
        
        buttons = browser.find_elements_by_xpath('//button[@data-test-id="date-picker-apply-button"]')
        for but in buttons:
            if but.text == 'Apply':
                but.click()
        
        print('Applied')
        
        time.sleep(3)
        
        scrolling(browser) 
        
        time.sleep(3)
        
        browser.find_element_by_tag_name('body').send_keys(webdriver.common.keys.Keys.CONTROL + webdriver.common.keys.Keys.HOME)
        
        time.sleep(3)
        
        scrolling(browser)
        
        time.sleep(3)
        
        browser.find_element_by_tag_name('body').send_keys(webdriver.common.keys.Keys.CONTROL + webdriver.common.keys.Keys.HOME)
        
        time.sleep(3)
        
        scrolling(browser)
        
        time.sleep(3)
        
        data = collect_data(browser, date_start)
        
        print('data collected')
        
        date_start = datetime.datetime(date_start.year, date_start.month, 1)
        
        with open(f'{file_loc}/{asset}/{date_start.year}-{date_start.month}', 'wb') as f:
            pickle.dump(data, f)
        
        browser.find_element_by_tag_name('body').send_keys(webdriver.common.keys.Keys.CONTROL + webdriver.common.keys.Keys.HOME)
        time.sleep(1)
        browser.find_element_by_tag_name('body').send_keys(webdriver.common.keys.Keys.CONTROL + webdriver.common.keys.Keys.HOME)
        
    except Exception as exc:
        print(exc)
        breakpoint()
        
