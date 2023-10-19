import pandas as pd
import numpy as np
import os

if os.path.isdir("./data") is False:
    os.mkdir("./data")

if os.path.isdir("./data/merged_data") is False:
    os.mkdir("./data/merged_data")

first_file_names = os.listdir('./data/first_data')
second_file_names = os.listdir('./data/second_data')

if len(first_file_names) == len(second_file_names):
    for i in range(len(first_file_names)):

        # 첫 번째 CSV 파일 로드
        df1 = pd.read_csv(f'./data/first_data/{first_file_names[i]}')
        df1 = df1[['Date', 'Close', '대비', '등락률', 'EPS', 'PER', 'BPS', 'PBR', '주당배당금', '배당수익률']]
        df1.columns = ['Date', 'Close', '대비', '등락률', 'EPS', 'PER', 'BPS', 'PBR', '주당배당금', '배당수익률']

        # 두 번째 CSV 파일 로드
        df2 = pd.read_csv(f'./data/second_data/{second_file_names[i]}', encoding='CP949')
        df2 = df2[['날짜', '거래량', '상장주식수']]
        df2['거래량'] = df2['거래량'].str.replace(',', '').astype(np.int64)
        df2['상장주식수'] = df2['상장주식수'].str.replace(',', '').astype(np.int64)
        df2['회전율'] = round(df2['거래량'] / df2['상장주식수'], 4)
        df2 = df2[['날짜', '회전율']]
        df2.columns = ['Date', '회전율']

        # df2.columns = ['Date', '회전율']

        df3 = pd.read_csv('./data/ExchangeRate.csv')
        df3 = df3[['Date', '종가']]
        df3.columns = ['Date', '환율']

        df4 = pd.read_csv('./data/KOSPI.csv', encoding='CP949')
        df4 = df4[['날짜', '종가']]
        df4.columns = ['Date', '코스피']
        df4['코스피'] = df4['코스피'].str.replace(',', '').astype(float)

        df5 = pd.read_csv('./data/NASDAQ.csv')
        df5 = df5[['Date', 'Close']]
        df5.columns = ['Date', '나스닥']

        nasdaq_list = df5['나스닥'].tolist()

        for j in range(len(df5)):
            fluctuate_rate = (nasdaq_list[j] - nasdaq_list[j - 1]) / nasdaq_list[j - 1] * 100
            df5.loc[j, '나스닥'] = round(fluctuate_rate, 2)

        # 날짜 컬럼을 datetime 형식으로 변환
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        df3['Date'] = pd.to_datetime(df3['Date'])
        df4['Date'] = pd.to_datetime(df4['Date'])
        df5['Date'] = pd.to_datetime(df5['Date'])

        # 날짜를 기준으로 두 데이터프레임을 병합
        merged_df = pd.merge(df1, df2, on='Date', how='inner')  # on : 어떤 값을 사용해서, how : 어떤 방식으로 join 할 것인지
        merged_df = pd.merge(merged_df, df3, on='Date', how='inner')
        merged_df = pd.merge(merged_df, df4, on='Date', how='inner')
        merged_df = pd.merge(merged_df, df5, on='Date', how='inner')

        # 날짜를 내림차순으로 정렬
        merged_df_sorted = merged_df.sort_values(by='Date', ascending=True)  # 오름차순으로 정렬하려면 ascending=True로 변경

        # 결과 출력
        merged_df_sorted.to_csv(f'./data/merged_data/{first_file_names[i]}', index=False)