import sys
sys.path.append('../')

from tqdm import tqdm
import pandas as pd
from utils import read_data
import numpy as np

target = 'pm25'
OUTPUT_NAME = 'aggregates_lag_one_deviations_and_temperature'


def add_rolling_aggregates(df, test_size, rolling_lag=None):
    if not rolling_lag:
        rolling_lag = test_size
    else:
        assert rolling_lag <= test_size

    for num in tqdm(range(1, rolling_lag + 1)):
        df[f'rolling_mean{num}'] = df[target].shift(test_size).rolling(num).mean()
        df[f'rolling_median{num}'] = df[target].shift(test_size).rolling(num).median()

        # seasonal
        df[f'rolling_mean_at_hour{num}'] = None
        for hour in tqdm(df['hour'].unique()):
            part_df = df[df['hour'] == hour]
            df.loc[df['hour'] == hour, f'rolling_mean_at_hour{num}'] = part_df[target].shift(1).rolling(num).mean()

    return df


def add_rolling_aggregates_lag_one(df, test_size=1, rolling_lag=10):
    for num in tqdm(range(1, rolling_lag + 1)):
        df[f'rolling_mean{num}'] = df[target].shift(test_size).rolling(num).mean()
        df[f'rolling_median{num}'] = df[target].shift(test_size).rolling(num).median()

    return df


def add_rolling_aggregates_lag_one_and_at_hour(df, test_size=1, rolling_lag=10):
    for num in tqdm(range(1, rolling_lag + 1)):
        df[f'rolling_mean{num}'] = df[target].shift(test_size).rolling(num).mean()
        df[f'rolling_median{num}'] = df[target].shift(test_size).rolling(num).median()

        df[f'rolling_mean_at_hour{num}'] = None
        for hour in tqdm(df['hour'].unique()):
            part_df = df[df['hour'] == hour]
            df.loc[df['hour'] == hour, f'rolling_mean_at_hour{num}'] = part_df[target].shift(1).rolling(num).mean()

    return df


def add_avg_deviation_from_previous(df):
    # how much does current hour deviate on avg from previous
    # first calculate avgs for all
    median_by_hour = df[[target, 'hour']].groupby(by='hour').median()

    for hour in range(24):
        df.loc[df['hour'] == hour, 'median_by_hour'] = median_by_hour.iloc[hour].values

    df['deviation_from_previous'] = np.NaN
    for hour in range(24):
        df.loc[df['hour'] == hour, 'deviation_from_previous'] = \
            df[df['hour'] == hour]['median_by_hour'].iloc[0] \
            - df[df['hour'] ==( hour - 1) % 24]['median_by_hour'].iloc[0]

    return df


def combined_deviations_and_simple_lags_one(df):
    df = add_rolling_aggregates_lag_one(df, rolling_lag=5)
    df = add_avg_deviation_from_previous(df)
    return df


def add_rolling_temperature_aggregate(df, feat='temperature', rolling_lag=5, test_size=24):
    for num in tqdm(range(1, rolling_lag + 1)):
        df[f'rolling_mean_{feat}_{num}'] = df[feat].shift(test_size).rolling(num).mean()

    return df


def combined_deviations_and_temperature(df):
    df = add_avg_deviation_from_previous(df)
    df = add_rolling_temperature_aggregate(df)
    return df


def combined_deviations_and_temperature_and_lag_one(df):
    df = add_avg_deviation_from_previous(df)
    df = add_rolling_temperature_aggregate(df)
    df = add_rolling_aggregates_lag_one(df, rolling_lag=5)
    return df


if __name__ == "__main__":
    train, test = read_data('../data_processed/period_id_added')

    agg_func = combined_deviations_and_temperature_and_lag_one

    trains = []
    tests = []
    combined = pd.concat([train, test]).sort_index()
    for idx in train['period_id'].unique():
        train_part = train[train['period_id'] == idx]
        test_part = combined[combined['period_id'] == idx]
        valid_size = len(train_part[train_part['split'] == 'valid'])

        # test_size = len(test_part[test_part['isTest']])
        # assert valid_size == test_size

        test_size = 1

        lags_to_produce = 10
        enriched_train = agg_func(train_part)

        enriched_test = agg_func(test_part)

        trains.append(enriched_train)
        tests.append(enriched_test)

    new_train = pd.concat(trains)
    new_test = pd.concat(tests)
    new_test = new_test[new_test['isTest']]

    assert new_train.index.is_monotonic_increasing
    assert new_train['id'].is_monotonic_increasing
    assert len(new_train) == len(train)
    assert new_test.index.is_monotonic_increasing
    assert new_test['id'].is_monotonic_increasing
    assert len(new_test) == len(test)

    new_train.to_hdf(f'../data_processed/{OUTPUT_NAME}/train.h5', key='train', mode='w')
    # new_test.to_hdf(f'../data_processed/{output_name}/test.h5', key='train', mode='w')
    new_test.to_hdf(f'../data_processed/{OUTPUT_NAME}/test.h5', key='train', mode='w')