import logging
import requests
from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_webpage(url):
    
    logging.getLogger('webdriver_manager').setLevel(logging.WARNING)

    headers = {'Accept-Language': 'en-US,en;q=0.5'}   
    prefs = {
      "translate_whitelists": {"your native language":"en"},
      "translate":{"enabled":"True"}
    }
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

    try:
        page = requests.get(url)
        driver.maximize_window()
        driver.get(url)

    except Exception as e:
        print(f"An error occurred 1: {e}")
        
    try:
        lan = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(("xpath",'//*[@id="modal"]/div[1]/div[1]/div/div[3]/div[2]/button')))
        if driver.find_element("xpath",'//*[@id="modal"]/div[1]/div[1]/div/div[3]/div[2]/button').get_attribute("inner_html") != "":
            driver.find_element("xpath",'//*[@id="modal"]/div[1]/div[1]/div/div[3]/div[2]/button').click()
    except Exception as e:
        print(f"An error occurred 2: {e}")
        
    sleep(5)
    
    try:
        option_top = driver.find_element('xpath',f"//*[contains(concat('',@class,''), 'shopee-sort-by-options')]/div[{3}]")
        option_top.click()
    except Exception as e:
        print(f"An error occurred 3: {e}")
    
    print('*** Shopee open successful ***')

    return driver

