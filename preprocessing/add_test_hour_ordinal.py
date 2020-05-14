import sys
sys.path.append('../')    # needed for preprocessing.sh to work correctly
import numpy as np
import pandas as pd

from utils import read_data

DATA_NAME = 'aggregates_lag_one_deviations_and_temperature'

def add_ordinals(df, period_length=24):
    df['ordinal'] = np.NaN
    for period_id_idx in df['period_id'].unique():
        period_df = df[df['period_id'] == period_id_idx]
        assert len(period_df) == period_length
        df.loc[df['period_id'] == period_id_idx, 'ordinal'] = np.arange(period_length)

    return df


train_and_valid, test = read_data(f'../data_processed/{DATA_NAME}', ext='h5')
train_and_valid.to_hdf(f'../data_processed/{DATA_NAME}/old_train.h5', key='train', mode='w')
test.to_hdf(f'../data_processed/{DATA_NAME}/old_test.h5', key='train', mode='w')

train_and_valid['ordinal'] = np.NaN
train = train_and_valid[train_and_valid['split'] == 'train']
valid = train_and_valid[train_and_valid['split'] == 'valid']
valid = add_ordinals(valid)

new_train = pd.concat([train, valid]).sort_index()
new_test = add_ordinals(test)

new_train.to_hdf(f'../data_processed/{DATA_NAME}/train.h5', key='train', mode='w')
new_test.to_hdf(f'../data_processed/{DATA_NAME}/test.h5', key='train', mode='w')