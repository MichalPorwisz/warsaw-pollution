import functools
import os

import numpy as np
import pandas as pd


def flatten(a_list):
    flat_list = []
    for sublist in a_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def save_submission(test, filename, dir='./output/', val_score=None):
    assert test['id'].nunique() == 1200

    if val_score:
        val_score_str = f'_{val_score:.1f}'
    else:
        val_score_str = ''

    path = os.path.join(dir, f'{filename}{val_score_str}')
    test = test[['id', 'pm25']]
    test['id'] = test['id'].astype(int)
    test.to_csv(path, index=False)
    print(f'saved to {path}')


def convert_type(df, col, type='string'):
    df.loc[:, col] = df[col].astype(type)


def read_data(dir, train_only=False, ext='csv'):
    if ext == 'csv':
        kwargs = { 'index_col': 0, 'parse_dates': [0]}
        load_method = functools.partial(pd.read_csv, **kwargs)
    elif ext == 'h5':
        load_method = pd.read_hdf
    else:
        raise Exception('not implemented')

    train = load_method(os.path.join(dir, f'train.{ext}'))
    assert train.index.is_monotonic_increasing

    if train_only:
        return train
    else:
        test = load_method(os.path.join(dir, f'test.{ext}'))
        assert test.index.is_monotonic_increasing
        dfs = tuple([train, test])

    return dfs


def convert_to_float_or_factorize_objects(df, feats):
    types_to_convert = df.dtypes[(df.dtypes == 'object') | (df.dtypes == 'string')]

    types_to_factorize = []
    for col in [col for col in types_to_convert.index if col in feats]:
        try:
            df.loc[:, col] = df[col].astype(np.float)
        except:
            types_to_factorize.append(col)


    for col in types_to_factorize:
        df.loc[:, col] = pd.factorize(df[col])[0]


    return df
