from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import StaleElementReferenceException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import os
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import requests
import pandas as pd

webdriver_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=webdriver_service)

if os.path.isdir("./data") is False:
    os.mkdir("./data")

if os.path.isdir("./data/first_data") is False:
    os.mkdir("./data/first_data")

def getCodeList():
    response = requests.get("https://finance.naver.com/sise/sise_market_sum.nhn")
    elements = BeautifulSoup(response.text, 'html.parser').find_all(class_="tltle")
    codes = ['005930', '000660', '000720', '005490', '036570'] # 삼성전자, SK하이닉스, 현대건설, POSCO홀딩스, 엔씨소프트
    return codes

HEADER = ['Date', 'Close', '대비', '등락률', 'EPS', 'PER', '선행 EPS', '선행 PER', 'BPS', 'PBR', '주당배당금', '배당수익률']
# => 선행 EPS, 선행 PER은 csv 읽어올 때 drop 하면 될듯

START_YEAR = 2014
END_YEAR = 2023
# => 여기서 학습용 / 테스트용 데이터로 분리해도 될듯

def match_code(code):
    if code == '005930':
        return '삼성전자'
    elif code == '000660':
        return 'SK하이닉스'
    elif code == '000720':
        return '현대건설'
    elif code == '005490':
        return 'POSCO홀딩스'
    else:
        return '엔씨소프트'

def get_data_rows(tbody):
    rows = tbody.find_elements(By.TAG_NAME, "tr")
    data = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        row_data = []
        for ele in cols:
            try:
                data_name = ele.get_attribute('data-name')
                if data_name in ['CMPPREVDD_PRC', 'FLUC_RT']:
                    span = ele.find_elements(By.XPATH, ".//span")
                    if span:
                        if data_name == 'CMPPREVDD_PRC' and "CI-GRID-UPDOWN-DOWN" in span[0].get_attribute('class'):
                            value = "-" + span[0].get_attribute('innerHTML')
                        else:
                            value = span[0].get_attribute('innerHTML')
                    else:
                        value = ele.get_attribute('innerHTML')
                else:
                    value = ele.get_attribute('innerHTML')
                row_data.append(value)
            except StaleElementReferenceException:
                pass
        data.append(row_data)
    return data

try:
    for code in getCodeList():
        # 쓰기 모드로 저장할 CSV 파일 생성 or 열기
        with open(f'./data/first_data/{match_code(code)}.csv', 'w', encoding='utf-8') as f:
            f.write(','.join(HEADER) + '\n')  # 생성한 CSV 파일에 헤더(features) 추가

            driver.get("http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020502")
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """ Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) """})
            searchType = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'searchType_0_1')))
            driver.execute_script("arguments[0].click();", searchType)
            finder = driver.find_element(By.ID, 'tboxisuCd_finder_stkisu0_0')
            startCalender = driver.find_element(By.ID, 'startCalender')
            endCalendar = driver.find_element(By.ID, 'endCalendar')
            searchButton = driver.find_element(By.ID, 'jsSearchButton')

            finder.clear()
            finder.send_keys(code)
            finder.send_keys(Keys.ENTER)
            while '/' not in str(finder.get_attribute('value')):
                sleep(1)

            for year in range(START_YEAR, END_YEAR + 1):

                for half in range(2):
                    startCalender.clear()
                    endCalendar.clear()
                    startCalender.send_keys('{0}{1}01'.format(year, '01' if half == 0 else '07'))
                    endCalendar.send_keys('{0}{1}{2}'.format(year, '06' if half == 0 else '12', '30' if half == 0 else '31'))
                    searchButton.click()
                    sleep(1)
                    tbody = driver.find_elements(By.CLASS_NAME, 'CI-GRID-BODY-TABLE-TBODY')[1]
                    data = get_data_rows(tbody)
                    data = reversed(data)

                    for row_data in data:
                        row_data = [data.replace(',', '') if ',' in data else data for data in row_data]
                        f.write(','.join(row_data) + '\n')  # CSV 파일에 쓰기
                    sleep(2)

finally:
    driver.quit()