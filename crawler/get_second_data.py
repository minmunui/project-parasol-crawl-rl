import os
import csv
import time
import requests
import datetime
import pandas as pd
import yfinance as yf

today = datetime.datetime.today()

if os.path.isdir("./data") is False:
    os.mkdir("./data")

if os.path.isdir("./data/second_data") is False:
    os.mkdir("./data/second_data")

print("환율 진행중...")
ExchangeRate: pd.DataFrame = yf.download(
    ["USDKRW=X"], start="2000-01-01", end=today.strftime("%Y-%m-%d")
)
ExchangeRate.drop("Volume", axis=1, inplace=True)
ExchangeRate.rename(
    columns={
        "Open": "시가",
        "High": "고가",
        "Low": "저가",
        "Close": "종가",
        "Adj Close": "수정종가",
    },
    inplace=True,
)

ExchangeRate.to_csv("data/ExchangeRate.csv", encoding="utf-8-sig")

print("나스닥 지수 진행중...")
NASDAQ: pd.DataFrame = yf.download(
    ["^IXIC"], start="2000-01-01", end=today.strftime("%Y-%m-%d")
)
NASDAQ.rename(
    columns={
        "Open": "시가",
        "High": "고가",
        "Low": "저가",
        "Close": "종가",
        "Adj Close": "수정종가",
        "Volume": "거래량",
    },
)
NASDAQ.to_csv("data/NASDAQ.csv", encoding="utf-8-sig")

print("KOSPI 지수 진행중...")

KOSPI_QUERY = [["날짜", "종가", "대비", "등락률", "시가", "고가", "저가", "거래량", "거래대금", "상장시가총액"]]

for year in range(2000, today.year + 1):
    KOSPI = requests.post(
        url="http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
        data={
            "bld": "dbms/MDC/STAT/standard/MDCSTAT00301",
            "tboxindIdx_finder_equidx0_1": "코스피",
            "indIdx": "1",
            "indIdx2": "001",
            "codeNmindIdx_finder_equidx0_1": "코스피",
            "strtDd": f"{year}0101",
            "endDd": today.strftime("%Y%m%d")
            if year == today.year
            else f"{year + 1}0101",
            "share": "1",
            "money": "1",
            "csvxls_isNo": "false",
        },
    ).json()

    for block in KOSPI["output"]:
        KOSPI_QUERY.append(
            [
                block["TRD_DD"],
                block["CLSPRC_IDX"],
                block["PRV_DD_CMPR"],
                block["UPDN_RATE"],
                block["OPNPRC_IDX"],
                block["HGPRC_IDX"],
                block["LWPRC_IDX"],
                ",".join(block["ACC_TRDVOL"].split(",")[:-1]),
                ",".join(block["ACC_TRDVAL"].split(",")[:-2]),
                ",".join(block["MKTCAP"].split(",")[:-2]),
            ]
        )
    time.sleep(1)

with open(f"data/KOSPI.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(KOSPI_QUERY)


financeCodes = ["005930", "000660", "000720", "005490", "036570"]

for code in financeCodes:
    codeInformation = (
        requests.post(
            url="http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            data={
                "locale": "ko_KR",
                "mktsel": "ALL",
                "typeNo": "0",
                "searchText": code,
                "bld": "dbms/comm/finder/finder_stkisu",
            },
        ).json()
    )["block1"][0]

    print(f'{codeInformation["codeName"]} 진행중...')

    ExchangeRate = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT01701",
        "locale": "ko_KR",
        "tboxisuCd_finder_stkisu0_0": code,
        "isuCd": codeInformation["full_code"],
        "param1isuCd_finder_stkisu0_0": "ALL",
        "strtDd": "19500101",
        "endDd": f"{today.strftime('%Y%m%d')}",
        "adjStkPrc_check": "Y",
        "adjStkPrc": "2",
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }

    stockInformation = requests.post(
        url="http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd", data=ExchangeRate
    ).json()

    query = [
        ["날짜", "종가", "대비", "등락률", "시가", "고가", "저가", "거래량", "거래대금", "시가총액", "상장주식수"]
    ]
    for block in stockInformation["output"]:
        query.append(
            [
                block["TRD_DD"],
                block["TDD_CLSPRC"],
                block["FLUC_TP_CD"],
                block["CMPPREVDD_PRC"],
                block["TDD_OPNPRC"],
                block["TDD_HGPRC"],
                block["TDD_LWPRC"],
                block["ACC_TRDVOL"],
                block["ACC_TRDVAL"],
                block["MKTCAP"],
                block["LIST_SHRS"],
            ]
        )

    with open(f"./data/second_data/{codeInformation['codeName']}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(query)
    time.sleep(5)
