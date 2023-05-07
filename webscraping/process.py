import logging
from selenium import webdriver
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ExpectError(Exception):
    def __init__(self, message="This error is expected."):
        self.message = 'something went wrong'
        super().__init__(self.message)

def extract_number(number):
    if "พัน" in number and "." in number:
        decimal_part = re.findall(r'[\d]*[.][\d]+', number)
        number = int(float(decimal_part[0]) * 1000)
    elif "พัน" in number and "." not in number:
        number = int(re.findall(r'\d+', number)[0]) * 1000
    else: 
        number = int(re.findall(r'\d+', number)[0])
    return number

def extract_number_1(number):
    if number == '0':
        return number
    elif "k" in number and "." in number:
        decimal_part = re.findall(r'[\d]*[.][\d]+', number)
        number = int(float(decimal_part[0]) * 1000)
    elif "k" in number and "." not in number:
        number = int(re.findall(r'\d+', number)[0]) * 1000
    else: 
        number = int(re.findall(r'\d+', number)[0])
    return number

def get_details(s):
    date = re.findall('\d{4}-\d{2}-\d{2}', s)[0]
    timing = re.findall('\d{2}:\d{2}', s)[0]
    num_like = re.search(r'\d+(?=Report Abuse)', s).group(0) if re.search(r'\d+(?=Report Abuse)', s) else 0
    return date, timing, num_like

def is_date_in_2022(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.year == 2023 or date_obj.year == 2022

def split_comment(comment):
    comment_1, comment_2 = comment.split(" , ")
    if not comment_2:
        return comment_1
    elif "Variation" in comment_1:
        return comment_2
    else:
        return comment_1

def play_song():
    file = os.path.join(os.path.join(os.getcwd(), "utility"), "alarm.mp3")
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()
        
    play_music = True

    while play_music:
        pygame.mixer.music.play()
        try:
            while pygame.mixer.music.get_busy():
                pass
        except KeyboardInterrupt:
            pygame.mixer.music.stop()
            play_music = False

def loop_product(driver, products):

    for product in range (1,products+1):
        print('number of product:', product)
        
        try:
            click = WebDriverWait(driver, 20).until(EC.element_to_be_clickable(('xpath',f"//*[contains(concat('',@class,''), 'row shopee-search-item-result__items')]/div[{product}]")))
            
            click = driver.find_element('xpath',f"//*[contains(concat('',@class,''), 'row shopee-search-item-result__items')]/div[{product}]")
            
            click.click()
            
        except:
            print(Fore.RED + 'Problem with product window' + Style.RESET_ALL)
            driver.implicitly_wait(1)

    # go back
        get_info(driver)
        
        sleep(1)
        driver.execute_script("window.history.go(-1)")
        sleep(1)
    
    return driver

def get_info(driver):

# ---------------------- get name ------------------------------- #  
   
    try:
        product_name = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(('xpath','//*[@id="main"]/div/div[2]/div[1]/div/div[1]/div[2]/div[2]/div[3]/div/div[1]/span')))
        product_name = driver.find_element('xpath','//*[@id="main"]/div/div[2]/div[1]/div/div[1]/div[2]/div[2]/div[3]/div/div[1]/span').get_attribute('textContent')
    except Exception as e:
        driver.implicitly_wait(1)
        print(Fore.RED + 'Problem with product name' + Style.RESET_ALL)
        raise ExpectError("This is an example of an expected error.")
        print(e)
        
    print(product_name)

# ---------------------- get url ------------------------------- # 
    try:
        current_url = driver.current_url
    except:
        driver.implicitly_wait(1)
        print(Fore.RED + 'Problem with product url' + Style.RESET_ALL)
    
# ---------------------- get rating ------------------------------- #

    try:
        overall_rate = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(('xpath','//*[@id="main"]/div/div[2]/div[1]/div[1]/div/div/div[2]/div[3]/div/div[2]/div[1]/div[1]')))
        overall_rate = driver.find_element('xpath','//*[@id="main"]/div/div[2]/div[1]/div[1]/div/div/div[2]/div[3]/div/div[2]/div[1]/div[1]').get_attribute('textContent')
    except:
        print(Fore.RED + 'Problem with rating' + Style.RESET_ALL)
        driver.implicitly_wait(1)

# ---------------------- get category ------------------------------- #
        
    try:
        text = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(('xpath',f"//*[contains(concat('',@class,''), 'flex items-center RnKf-X')]/a")))
        text = driver.find_elements('xpath',f"//*[contains(concat('',@class,''), 'flex items-center RnKf-X')]/a")

        cat_1 = text[-3].text
        cat_2 = text[-2].text
        cat_3 = text[-1].text

    except:
        print(Fore.RED + 'Problem with category' + Style.RESET_ALL)
        driver.implicitly_wait(1)
        
    full_list = [current_url, product_name, overall_rate, cat_1, cat_2, cat_3]
    loop_star(driver, full_list)
    
# ----------------------------------------------------- #

def loop_star(driver, full_list):
    bound = 2
    num_rate = [] # 5,4,3,2,1
    for index in range(5): # 2-7 # 2=5star, 6=1star
        index += 2
        current_star = 6 - (index - 1)
        
        try:
            with_ment = WebDriverWait(driver, 14).until(EC.element_to_be_clickable(('xpath', f'//*[@id="main"]/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[2]/div[2]/div[{index}]')))
            with_ment = driver.find_element('xpath', f'//*[@id="main"]/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[2]/div[2]/div[{index}]')

            with_ment.click()
            sleep(2)
            with_ment.click()
        except:
            print(Fore.RED + 'Problem with clicking to see comment' + Style.RESET_ALL)
            driver.implicitly_wait(1)

        
        number_rate = extract_number(with_ment.get_attribute('textContent'))
        
        num_rate.append(number_rate)
        
        if number_rate == 0:
            continue
        
        ### get comment here
        
        # go for review page
        for curr_page in range(3, 3+(bound-1)): # 2 is page 1
            comment_page_path1 = f'//*[@id="main"]/div/div[2]/div[1]/div/div/div[2]/div[3]/div[2]/div[1]/div[2]/div/div/div[3]/div[2]/button[{curr_page}]'
            comment_page_path2 = f'//*[@id="main"]/div/div[2]/div[1]/div/div/div[2]/div[4]/div[2]/div[1]/div[2]/div/div/div[3]/div[2]/button[{curr_page}]'
            # print(comment_page_path)
            
            try:
                if bool(driver.find_element('xpath',comment_page_path1)):
                    cl = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(('xpath', comment_page_path1)))
                    cl.click()
                    sleep(2)
            except NoSuchElementException:
                try:
                    if bool(driver.find_element('xpath',comment_page_path2)):
                        cl = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(('xpath', comment_page_path2)))
                        cl.click()
                        sleep(2)
                except:
                    break # go to next star sampling
            except:
                break # go to next star sampling
                                    
            content = loop_review(driver, full_list)
 
        ###
                
    sleep(2)
    driver.execute_script("window.history.go(-1)")
    sleep(1)

    return None

def loop_review(driver, full_list):
    global fin
    global log
    for index in range(1, 7):

        try:
            text1 = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/*/div[3]")))
            text1 = driver.find_element('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/*/div[3]").get_attribute("textContent")
        except Exception as e:
            try:
                text3 = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/div[3]")))
                text3 = driver.find_element('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/div[3]").get_attribute("textContent")                
                text2 = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/div[4]")))
                text2 = driver.find_element('xpath',f"//*[@id='main']/div/div[2]/div[1]/div/*/div[2]/*/*/div[1]/div[2]/div/div/div[3]/div[1]/div[{index}]/*/div[4]").get_attribute("textContent")
                text1 = text3 + " , " + text2
            except Exception as e:
                print(Fore.YELLOW + 'Problem with comment' + Style.RESET_ALL)
                break
        try:
            comment = split_comment(text1)
        except Exception as e:
            comment = text1
        
        if 'helpful?Report Abuse' in comment or 'response' in comment or len(text1) < 4:
            print(Fore.YELLOW + 'No comment found' + Style.RESET_ALL)
            break

        try:
            details = WebDriverWait(driver, 1.5).until(EC.element_to_be_clickable(('xpath',f"//*[contains(concat('',@class,''), 'shopee-product-comment-list')]/div[{index}]")))
            details = driver.find_element('xpath',f"//*[contains(concat('',@class,''), 'shopee-product-comment-list')]/div[{index}]").get_attribute('textContent')
            a, b, c = get_details(details)
            date = a
            timing = b
            num_like = c
        except:
            print(Fore.YELLOW + 'Problem with details' + Style.RESET_ALL)
            continue
            
        if not is_date_in_2022(date):
            print(Fore.YELLOW + 'Date exceed' + Style.RESET_ALL)
            continue

        try:
            for rating in driver.find_elements("css selector",".shopee-product-rating__rating"):
                stars = rating.find_elements("css selector",".icon-rating-solid")
                stars = len(stars)
        except:
            print(Fore.YELLOW + 'problem with review star' + Style.RESET_ALL)
            continue
        
        row = full_list + [date, timing, num_like, stars, comment]
        log += 1
        print(str(log) + ': ' + comment)
        fin.append(row)