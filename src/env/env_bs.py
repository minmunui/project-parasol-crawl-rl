from enum import Enum
from typing import Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType

DEFAULT_OPTION = {
    'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': 500,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
    'commission': .0003,  # 수수료
    'selling_tax': .00015,  # 매도세
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}


class MyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, option=DEFAULT_OPTION):
        """
        환경을 구축합니다. 여기서 환경이란 주식 데이터와 환경 옵션을 의미합니다.

        df는 주식 데이터가 들어있는 DataFrame입니다. 주식 데이터는 일자를 인덱스로 가지며, 종가('closed')를 반드시 가지고 있어야 합니다.

        각 df는 Date, Close를 반드시 가지고 있어야 합니다.
        Date는 날짜를 의미하며, Close는 종가를 의미합니다.

        모든 구매, 판매는 종가(Close)를 기준으로 이루어집니다.

        options는 환경 옵션을 의미합니다. 자세한 내용은 default_options를 참고하세요.
        :param df: 주식 데이터가 들어있는 DataFrame의 배열
        :param option: 환경 옵션
        """

        print(len(df))
        assert 'Close' in df.columns, "df must have 'Close' column"
        assert option['start_index'] >= 0, "start_index must be greater than 0"
        assert option['end_index'] <= len(df), "end_index must be less than len(df)"
        assert option['start_index'] < option['end_index'], "start_index must be less than end_index"
        assert len(df) > option['window_size'] > 0, "window_size must be greater than 0 and less than len(df)"
        assert option['proportion_precision'] > 0, "proportion_precision must be greater than 0"

        # 주식의 데이터 수를 저장
        self.df = df
        self.number_of_properties = len(df.columns)

        self.option = {key: option.get(key, value) for key, value in DEFAULT_OPTION.items()}

        # Agent가 획득하는 데이터의 형태 (감시할 기간, 감시할 데이터의 수)
        self.shape = (option['window_size'], self.number_of_properties)

        # 수수료와 세금 세금은 매도할 때만 발생
        self.selling_tax = self.option['selling_tax']
        self.commission = self.option['commission']
        self.proportion_precision = self.option['proportion_precision']
        self.window_size = self.option['window_size']
        self.reward_threshold = self.option['reward_threshold']

        # 초기 자본금 = 최대종가 * 비율정밀도 * 계수 (1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함)
        self.init_balance = self.option['initial_balance_coef'] * df['Close'].max() * self.option[
            'proportion_precision']

        # 시작과 끝 그리고 현재 index
        self.start_index = self.option['start_index'] + self.window_size - 1
        self.end_index = len(df) - 1
        self._current_index = 0

        # 현재 보유 현금과, 현재 포트폴리오 가치
        self._balance = None
        self._total_value = None

        # 보유 주식 비율, 보유 주식 수, 평균 매수가e
        self._current_proportion = None
        self._holdings = None
        self._avg_buy_price = None

        # 수익률은 1을 기준으로 함
        self._profit = None
        self._reward_huddle = None

        # 현재 상태와 가격, 주가데이터
        self._state = {}
        self.price = df['Close'].values.astype(np.float32)
        self._stock_data = None

        # action을 저장할 변수
        self._history = []

        # 종료 여부
        self._done = False
        # 가능한 Action은 보유비율 증가, 유지, 감소
        self.action_space = gym.spaces.Discrete(3)

        # 가능한 Observation은 각 종목별 현재가, 보유량, 평균 매수가, 현금
        self.observation_space = gym.spaces.Dict({
            # 보유 주식 비율
            "holding": gym.spaces.Discrete(self.option['proportion_precision'] + 1),

            # 평균 매수가 : 최대 가격을 넘을 수 없음
            "avg_buy_price": gym.spaces.Box(
                low=0,
                high=np.inf,
                dtype=np.float32
            ),

            # 주가데이터는 종목의 수 * 사용할 데이터 수
            "data": gym.spaces.Box(
                low=np.inf,
                high=np.inf,
                shape=self.shape,
                dtype=np.float32
            )
        })
        # 환경 초기화
        self.reset()

    def reset(self) -> ObsType:
        # 초기화
        self._holdings = 0
        self._avg_buy_price = 0
        self._current_proportion = 0
        self._current_proportion = np.clip(self._current_proportion, 0, self.proportion_precision)
        self._profit = 1.0
        self._total_value = self.init_balance
        self._balance = self.init_balance
        self._current_index = self.start_index
        self._reward_huddle = self.init_balance
        self._history = []
        self._done = False
        self._state = {}
        # 초기 상태 반환
        return self._observe()

    def _observe(self) -> ObsType:

        if self._current_index >= self.end_index:
            self._done = True
            return self._state

        # 각 종목의 데이터를 가져와서 stock_info, self.prices에 추가
        self._stock_data = self.df[(self._current_index - self.window_size + 1):self._current_index + 1].values
        # 각종 정보를 state에 추가
        self._state['data'] = self._stock_data
        self._state['holding'] = self._current_proportion
        # 이전의 환경과 비슷한 느낌?
        # self._state['avg_buy_price'] = self._avg_buy_price
        self._state['avg_buy_price'] = np.array([self._avg_buy_price], dtype=np.float32)
        return self._state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        # 종료 여부 확인
        if self._done:
            raise Exception("Episode is done")

        # print('action = {0}'.format({action}))
        # 매수
        if action == 0:
            if not (self._current_proportion == self.proportion_precision):
                # 매수량 결정하기
                self._current_proportion += 1
                amount = int(self._get_hold_amount() - self._holdings)
                # print('amount_to_buy : {0}'.format(amount))
                self._buy(amount)

        # 매도
        elif action == 1:
            if not (self._current_proportion == 0):
                # 매도량 결정하기
                self._current_proportion -= 1
                amount = int(self._holdings - self._get_hold_amount())
                self._sell(amount)


        # action 후 기록
        info = dict(
            index=self._current_index,
            total_value=self._total_value,
            profit=self._profit,
            action=action
        )
        self._update_history(info)

        # 다음 index로 넘어가기
        self._current_index += 1

        self._get_total_value()
        self._get_profit()
        self._observe()
        reward = self._get_reward()

        return self._state, reward, self._done, info

    def _update_history(self, info):
        # history에 info 추가
        self._history.append(info)

    def _get_hold_amount(self) -> int:
        # 최대 보유 가능 주식 수 계산
        return int(self._total_value * self._current_proportion /
                   self.proportion_precision /
                   (self.price[self._current_index] * (1.0 + self.commission)))

    def _buy(self, amount: int) -> None:
        # 매수할 주식의 가격과 수수료 계산 후 잔고에서 차감 후 보유량 증가 평균 매수가 계산
        price = self.price[self._current_index]
        commission = price * amount * self.commission
        self._balance -= price * amount + commission
        self._holdings += amount
        if self._holdings > 0:
            self._avg_buy_price = (self._avg_buy_price * (self._holdings - amount) + price * amount) / self._holdings
        else:
            self._avg_buy_price = self.price[self._current_index]

    def _sell(self, amount: int) -> None:
        # 매도할 주식의 가격과 수수료,세금 계산 후 잔고에 반영, 보유량 감소
        price = self.price[self._current_index]
        commission = price * amount * self.commission
        tax = price * amount * self.selling_tax
        self._balance += price * amount - commission - tax
        self._holdings -= amount

    def _get_reward(self) -> float:
        # 보상은 수익이 reward_huddle의 +- {reward_threshold}% 범위에 들면 0, 아니면 수익의 +- /{reward_threshold}% 만큼의 보상
        reward_coefficient = self._reward_huddle * self.reward_threshold
        positive_threshold = self._reward_huddle + reward_coefficient
        negative_threshold = self._reward_huddle - reward_coefficient
        if self._total_value > positive_threshold or self._total_value < negative_threshold:
            reward = (self._total_value - self._reward_huddle) / reward_coefficient
            self._reward_huddle = self._total_value
            # print("reward : ", reward)
            # print("reward_huddle", self._reward_huddle)
            return reward
        else:
            return 0

    def _get_profit(self) -> float:
        self._profit = (self._total_value - self.init_balance) / self.init_balance + 1.0
        return self._profit

    def _get_total_value(self) -> float:
        self._total_value = self._balance + self._holdings * self.price[self._current_index]
        return self._total_value

    def _get_done(self) -> bool:
        self._done = self._current_index >= self.end_index
        return self._done

    def print_current_state(self):
        print("current_index : {}".format(self._current_index))
        print("price_last : {}".format(self.price[self._current_index-1]))
        print("price_current : {}".format(self.price[self._current_index]))
        print("holdings : {}".format(self._holdings))
        print("balance: {}".format(self._balance))
        print("total_value : {}".format(self._total_value))
        print("total_profit : {}".format(self._profit))
        print("current_holding_proportion : {}".format(self._current_proportion))
