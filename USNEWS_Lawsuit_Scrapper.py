from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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

driver.get('https://www.usnews.com/topics/subjects/lawsuits')

container = driver.find_element_by_xpath('//div[@class="LoadMoreWrapper__Container-zwyk5c-0 himujt"]')
container = container.find_element_by_tag_name('div')

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
    time.sleep(1)
    scrolling(driver)
    buttons = driver.find_elements_by_tag_name('button')
    for but in buttons:
        if but.text == 'Load More':
            but.click()
            time.sleep(2)
    
    with open(f'USNEWS_Business', 'wb') as f:
        pickle.dump(driver.page_source, f)



"""
##Scroll to end

divs = container.find_elements_by_xpath('//div[@class="Box-w0dun1-0 MediaObject__Container-sc-19vl09d-0 bxtMxG eYkAGe story"]')

data = {'Date':[],
        'Asset':[],
        'Headline':[],
        'Author':[],
        'Href':[]}
count = 0
for div in divs:
    try:
        headline = div.find_element_by_tag_name('h3')
        href = headline.find_element_by_tag_name('a').get_attribute('href')
        headline = headline.text
        p = div.find_elements_by_tag_name('p')[1]
        author_date = p.find_elements_by_tag_name('span')
        author = author_date[0].text
        date = author_date[3].text
        date = datetime.datetime.strptime(date, '%b. %d, %Y')
        date = date.strftime('%Y-%m-%d')
        asset = 'usnews'
        
        data['Date'].append(date)
        data['Asset'].append(asset)
        data['Headline'].append(headline)
        data['Author'].append(author)
        data['Href'].append(href)
    except:
        count += 1
        print(f'Failed Count: {count}')
        breakpoint()
        
df = pd.DataFrame(data)

df.to_csv('USNEWS.csv', index=False)
"""