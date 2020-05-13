import sys

import pandas as pd

sys.path.append('../')
from utils import read_data

DATA_NAME = 'aggregates_lag_one_deviations_and_temperature'
NEW_DATA_NAME='forecasts_with_aggregates_lag_one_deviations_and_temperature'

train_and_valid, test = read_data(f'../data_processed/{DATA_NAME}', ext='h5')

# TODO: Could see also other columns they have because data retrieving filtered some (but most intersting stuff is there)
forecasts_df = pd.read_csv('../data_processed/forecasts/warsaw_full.csv', parse_dates=['date_time'], index_col='date_time')
for col in forecasts_df:
    forecasts_df.rename(columns={col: f'forecasted_{col}'}, inplace=True)

def add_forecasts(df, forecasts_df):

    return pd.merge(df, forecasts_df, how='left', left_index=True, right_index=True)

new_train = add_forecasts(train_and_valid, forecasts_df)
new_test = add_forecasts(test, forecasts_df)

new_train.to_hdf(f'../data_processed/{NEW_DATA_NAME}/train.h5', key='train', mode='w')
new_test.to_hdf(f'../data_processed/{NEW_DATA_NAME}/test.h5', key='train', mode='w')

print(forecasts_df.columns)