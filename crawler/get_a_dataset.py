import pandas as pd
import numpy as np
import os

if os.path.isdir("../data") is False:
    os.mkdir("../data")

if os.path.isdir("../data/A_dataset") is False:
    os.mkdir("../data/A_dataset")

file_names = os.listdir('./data/first_data')

for file in file_names:

    # CSV 파일 로드
    df = pd.read_csv(f'./data/first_data/{file}')
    df = df[['Date', 'Close', '대비', '등락률', 'BPS', 'PBR', '주당배당금', '배당수익률']]

    df.to_csv(f'../data/A_dataset/{file}', index=False)