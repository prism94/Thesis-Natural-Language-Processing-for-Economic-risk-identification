from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as bs
import pandas as pd
import datetime
import time
import pickle

press = 100

options = webdriver.ChromeOptions() 
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome('chromedriver.exe', options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})

driver.get('https://topclassactions.com/category/lawsuit-settlements/lawsuit-news/')

code = bs(driver.page_source, 'lxml')

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

for i in range(press):
    scrolling(driver)
    buttons = driver.find_elements_by_tag_name('button')
    for but in buttons:
        if but.text == 'Load More':
            but.click()
            time.sleep(2)

with open(f'topclassactions', 'wb') as f:
    pickle.dump(driver.page_source, f)

