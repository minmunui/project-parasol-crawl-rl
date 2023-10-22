# project-parasol-crawl-rl

<img alt="python" src ="https://img.shields.io/badge/python-3.10.12-3776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/>  <img alt="anaconda" src ="https://img.shields.io/badge/anaconda-23.3.1-3776AB.svg?&style=for-the-badge&logo=anaconda&logoColor=white"/>  <img alt="pyTorch" src ="https://img.shields.io/badge/pyTorch-2.1.0-3776AB.svg?&style=for-the-badge&logo=pyTorch&logoColor=white"/>


## 프로젝트 소개

웹 크롤링을 통해 데이터를 수집하고 수집한 데이터를 이용하여 강화학습 모델을 생성, 예측을 생성하는 프로젝트입니다.
<img width="608" alt="image" src="https://github.com/minmunui/project-parasol-crawl-rl/assets/82745129/764b18c9-c35a-46de-b613-d04578fd971a">


## 프로젝트 구조

```angular2html
project-parasol-crawl-rl
├── README.md
├── data    // 학습에 사용할 데이터
├── logs    // 학습 로그
├── models  // 생성된 모델
├── src
│   ├── env // 강화학습 환경
│   └── utils // 유틸리티
├── train_rl.py // 강화학습 모델 생성
└── test_rl.py // 강화학습 모델 예측
```



## 프로젝트 설치

해당 프로젝트는 `conda`를 이용하여 환경을 구성합니다.

#### 개발 환경
```angular2html
conda 23.3.1
python 3.10.12
```

### 환경 설정
#### 1. conda 환경 생성


```bash
conda create -n parasol python=3.10
```

#### 2. conda 환경 활성화

```bash
conda activate parasol
```

#### 3. conda 환경에 패키지 설치

```bash
pip install -r requirements.txt
```

### 데이터 수집
`crawler` 디렉토리에서 아래의 명령어를 순서대로 실행하여 크롤링을 진행합니다.
```bash
python ./get_first_data.py
python ./get_second_data.py
python ./get_merged_data.py
python ./get_a_dataset.py
python ./get_b_dataset.py
```



### 강화학습 모델 생성

#### 4. 수집된 데이터를 이용하여 강화학습 모델 생성
`root directory`에서 아래의 명령어를 실행합니다.
```bash
python ./train_rl.py --stock sk_hynix --algo dqn --lr 0.0001 --bs 256 --rb 1000000 --data_column Date Close 대비 등락률 BPS PBR 주당배당금 배당수익률 --timesteps 1000000 --data_path ./data/A_dataset/SK하이닉스.csv --start_date 2014/01/02 --end_date 2021/12/30
```
위의 속성은 조정이 가능 합니다. 자세한 내용은 아래와 같습니다.
```
--stock : 주식 종목명 -> 해당 모델의 이름이 됩니다.
--algo : 강화학습 알고리즘 (dqn, a2c)
--lr : 학습률
--bs : 배치 사이즈 (DQN의 경우)
--rb : 리플레이 버퍼 사이즈 (DQN의 경우)
--n_steps : 학습 횟수, 몇 step마다 업데이트 할 것인지 (A2C의 경우)
--data_column : 데이터 컬럼명
--timesteps : 학습 횟수
--data_path : 데이터 경로
--start_date : 학습 시작 날짜
--end_date : 학습 종료 날짜
```

사용이 가능한 데이터 컬럼은 아래와 같습니다.

```angular2html
Date,Close,대비,등락률,EPS,PER,BPS,PBR,주당배당금,배당수익률,회전율,환율,코스피,나스닥
```

#### 5. 생성된 강화학습 모델을 이용하여 예측

```bash
python test_rl_graph.py --stock sk_hynix --holding 0 --algo dqn --data_column Date Close 대비 등락률 BPS PBR 주당배당금 배당수익률 --model_path ./models/sk_hynix/dqn/dqn_0.0001_1000000_256.zip --data_path ./data/A_dataset/SK하이닉스.csv --start_date 2022/01/03 --end_date 2023/10/19 --graph True
```

```
--stock : 주식 종목명
--holding : 보유 주식 초기 chunk
--algo : 강화학습 알고리즘 (dqn, a2c)
--data_column : 데이터 컬럼명, 학습에 사용한 데이터 컬럼과 동일해야 합니다.
--model_path : 모델 경로
--data_path : 데이터 경로
--start_date : 학습 시작 날짜
--end_date : 학습 종료 날짜
--graph : 그래프 생성 여부 (True, False), 해당 모델의 성능을 확인할 수 있는 그래프를 생성합니다.
```
#### 6. test 로그 확인

`data/`에 생성된 로그를 확인합니다. 로그는 `csv` 파일로 생성됩니다.

0은 매수, 1은 매도, 2는 관망을 의미합니다.

결과 그래프는 `images/`에 생성됩니다.
