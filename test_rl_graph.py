import src.env.env_bs as env

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3 import A2C

import os
import re
import csv
import argparse
from datetime import datetime, timedelta

def get_index(df, date):
    index = df.loc[df.Date == date].index[0]

    return index

def get_date(df, index):
    date_obj = df.loc[index]['Date'].date()
    date_str = date_obj.strftime('%Y-%m-%d')

    return date_str

def load_data(data_path, args):
    df = pd.read_csv(data_path, na_values='-')

    df['Date'] = pd.to_datetime(df['Date'])

    start_index = get_index(df, args.start_date)
    end_index = get_index(df, args.end_date)

    df = df[start_index - 19:end_index + 1]

    df_list = args.data_column

    df = df[df_list]

    return df

def get_action(model, args): # 날짜를 주면?

    datas = []
    profits = []

    ref_stock = load_data(args.data_path, args)
    use_stock = ref_stock.drop('Date', axis=1, inplace=False)

    DEFAULT_OPTION = {
        'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
        'start_index': 0,  # 학습 시작 인덱스
        'end_index': len(use_stock) - 1,  # 학습 종료 인덱스
        'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
        'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
        'commission': .0003,  # 수수료
        'selling_tax': .00015,  # 매도세
        'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정 학습이 잘 되지 않는다면 이것을 건드려 보는 것을 추천 => 리워드 그래프가 이상할 때(너무 변동이 없을 때)
    }

    window_size = DEFAULT_OPTION['window_size']

    global env
    env = env.MyEnv(use_stock, option=DEFAULT_OPTION)

    obs = {'stock_data': use_stock.head(DEFAULT_OPTION['window_size']), 'holding': args.holding, 'avg_buy_price': np.array([0.], dtype=np.float32)}

    if args.algo == 'a2c':
        model = A2C.load(args.model_path, env=env)
    elif args.algo == 'dqn':
        model = DQN.load(args.model_path, env=env)

    start_index = get_index(ref_stock, args.start_date)
    end_index = get_index(ref_stock, args.end_date)

    iteration = end_index - start_index

    if start_index == end_index:
        date = get_date(ref_stock, start_index)
        close = get_close(ref_stock, date)

        action, _ = model.predict(obs, deterministic=True)

        datas.append({'stock_date': date, 'close': close, 'action': action.item()})
    else:
        for i in range(start_index, end_index):
            date = get_date(ref_stock, i)
            close = get_close(ref_stock, date)

            action, _ = model.predict(obs, deterministic=True)

            datas.append({'date': date, 'close': close, 'action': action.item()})

            obs, reward, done, info = env.step(action)

            profits.append(info['profit'])

            if i == end_index - 1:
                date = get_date(ref_stock, i + 1)
                close = get_close(ref_stock, date)

                action, _ = model.predict(obs, deterministic=True)

                datas.append({'date': date, 'close': close, 'action': action.item()})

    df = pd.DataFrame(datas)

    if not os.path.exists(f'./data'):
        os.makedirs(f'./data')

    df.to_csv(f'./data/{get_code(args.stock)}_{args.algo}.csv', index=False)

    if args.graph == True:
        closes = (ref_stock['Close'] / ref_stock['Close'].iloc[0]).values  # close feature를 초기 close 값으로 나눔
        plt.plot(closes, label='closes')

        for i in range(0, iteration):
            if datas[i]['action'] == 0:
                plt.scatter(i + window_size - 1, closes[i + window_size - 1], color='red')  # 실질적인 행동은 window_size만큼 밀린 종가에 대한 것
            elif datas[i]['action'] == 1:
                plt.scatter(i + window_size - 1, closes[i + window_size - 1], color='blue')

        x = np.arange(window_size, iteration + 20)
        print(x.shape)
        plt.plot(x, profits, label='profits')

        plt.xlabel('Episode')
        plt.ylabel('normalized closes & profits')
        plt.title('Profit Graph')
        plt.legend()

        if not os.path.exists(f'./images'):
            os.makedirs(f'./images')

        plt.savefig(f"./images/{args.stock}.png")

        plt.show()
    return datas

def get_action_prob(data_path):
    # 주어진 문자열
    action_probs = []

    df = pd.read_csv(data_path)

    datas = df.tail(1).values # data[0][0] / data[0][1] / data[0][2] 각각 매수 매도 관망

    for data in datas[0]:
        matches = re.findall(r'\d+\.\d+', data)

        if matches:
            number_str = matches[0]
            number = float(number_str)
            number = round(100 * number, 2)
            action_probs.append(number)
        else:
            print("숫자를 찾을 수 없습니다.")

    return action_probs


def get_close(stock_df, date):
    close = stock_df.loc[stock_df.Date == date].Close.values[0]

    return close

def get_stock_name(code):
    if code == '005930':
        return '삼성전자'
    elif code == '000660':
        return 'SK하이닉스'
    elif code == '000720':
        return '현대건설'
    elif code == '005490':
        return 'POSCO홀딩스'

def get_code(stock):
    if stock == 'samsung':
        return '005930'
    elif stock == 'sk_hynix':
        return '000660'
    elif stock == 'hyundai':
        return '000720'
    elif stock == 'posco':
        return '005490'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, choices=['samsung', 'hyundai', 'sk_hynix', 'posco'], required=True)  # 학습할 주식 종목
    parser.add_argument('--holding', type=int, choices=[0, 1, 2, 3, 4], required=True) # 보유 주식 비율
    parser.add_argument('--algo', type=str, choices=['a2c', 'dqn'], required=True)  # A2C or DQN
    parser.add_argument('--data_column', type=str, nargs='+', required=True)  # 데이터의 형태가 정해지면 그 때 choices 추가 / nargs='+'로 list 형태 입력 가능
    parser.add_argument('--model_path', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)  # 주가 데이터 파일(csv)이 위치한 경로
    parser.add_argument('--start_date', action='store', type=str, required=True)
    parser.add_argument('--end_date', action='store', type=str, required=True)
    parser.add_argument('--graph', action='store', type=bool, choices=[True], required=False)

    args = parser.parse_args()

    stock_names = args.stock

    get_action(args.model_path, args)
