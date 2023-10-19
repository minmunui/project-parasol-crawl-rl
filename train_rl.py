import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta

import src.env.env_bs as env

from stable_baselines3 import DQN
from stable_baselines3 import A2C

def get_index(df, date):
    index = df.loc[df.Date == date].index[0]

    return index

def load_data(data_path, args):
    df = pd.read_csv(data_path, na_values='-')

    df['Date'] = pd.to_datetime(df['Date'])

    start_index = get_index(df, args.start_date)
    end_index = get_index(df, args.end_date)

    df = df[start_index : end_index + 1]

    df_list = args.data_column

    df = df[df_list]

    df.drop('Date', axis=1, inplace=True)

    return df


def train(stock_names, args):
    if not os.path.exists('./models'):
        os.makedirs('./models')

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    stock_info = load_data(args.data_path, args)

    DEFAULT_OPTION = {
        'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
        'start_index': 0,  # 학습 시작 인덱스
        'end_index': len(stock_info) - 1,  # 학습 종료 인덱스
        'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
        'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
        'commission': .0003,  # 수수료
        'selling_tax': .00015,  # 매도세
        'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정 학습이 잘 되지 않는다면 이것을 건드려 보는 것을 추천 => 리워드 그래프가 이상할 때(너무 변동이 없을 때)
    }

    global env
    env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

    for stock_name in stock_names:

        if not os.path.exists(f'./models/{stock_name}'):
            os.makedirs(f'./models/{stock_name}')

        if not os.path.exists(f'./models/{stock_name}/{args.algo}'):
            os.makedirs(f'./models/{stock_name}/{args.algo}')

        if not os.path.exists(f'./logs/{stock_name}'):
            os.makedirs(f'./logs/{stock_name}')

        if args.algo == 'a2c':
            for lr in args.lr:
                for n_steps in args.n_steps:
                    model = A2C('MultiInputPolicy', env=env, learning_rate=lr, n_steps=n_steps, tensorboard_log=f'./logs/{stock_name}')
                    model.learn(total_timesteps=args.timesteps, progress_bar=True, tb_log_name=f'./logs/{stock_name}/{args.algo}_{lr}_{n_steps}')
                    model.save(f"./models/{stock_name}/{args.algo}/{args.algo}_{lr}_{n_steps}.zip")

        if args.algo == 'dqn':
            for lr in args.lr:
                for rb in args.rb:
                    for bs in args.bs:
                        model = DQN('MultiInputPolicy', env=env, learning_rate=lr, buffer_size=rb, batch_size=bs, tensorboard_log=f'./logs/{stock_name}')
                        model.learn(total_timesteps=args.timesteps, log_interval=1, progress_bar=True, tb_log_name=f'{args.algo}_{lr}_{rb}_{bs}')
                        model.save(f"./models/{stock_name}/{args.algo}/{args.algo}_{lr}_{rb}_{bs}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', type=str, choices=['samsung', 'hyundai', 'sk_hynix', 'posco'], nargs='+', required=True) # 학습할 주식 종목
    parser.add_argument('--algo', type=str, choices=['a2c', 'dqn'], required=True) # A2C or DQN
    parser.add_argument('--lr', action='store', type=float, nargs='+', required=True) # learning rate
    parser.add_argument('--bs', action='store', type=int, nargs='+', required=False)  # batch size for DQN
    parser.add_argument('--rb', action='store', type=int, nargs='+', required=False)  # replay buffer size for DQN
    parser.add_argument('--n_steps', action='store', type=int, nargs='+', required=False) # n steps for A2C
    parser.add_argument('--data_column', type=str, nargs='+', required=True) # 데이터의 형태가 정해지면 그 때 choices 추가 / nargs='+'로 list 형태 입력 가능
    parser.add_argument('--timesteps', action='store', type=int, required=True) # timesteps for reinforcement learning
    parser.add_argument('--data_path', action='store', type=str, required=True) # 주가 데이터 파일(csv)이 위치한 경로
    parser.add_argument('--start_date', action='store', type=str, required=True)
    parser.add_argument('--end_date', action='store', type=str, required=True)

    args = parser.parse_args()

    if args.algo == 'A2C':
        if args.n_steps is None:
            parser.error("--n_steps is required for A2C")
    elif args.algo == 'DQN':
        if args.bs is None or args.rb is None:
            parser.error("--bs and --rb are required for DQN")

    stock_names = args.stock

    train(stock_names, args)