import pandas as pd

def load_data(data_url):
    df = pd.read_csv(data_url, na_values='-')
    drop_list = ['Date']

    for drop in drop_list:
        df.drop(drop, axis=1, inplace=True)

    return df
