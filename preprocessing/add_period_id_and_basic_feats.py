import pandas as pd

train: pd.DataFrame = pd.read_hdf('../input/train_warsaw.h5')
test: pd.DataFrame = pd.read_hdf('../input/test_warsaw.h5')

target = 'pm25'


def preprocess(df):
    df['date'] = df.index.map(lambda x: x.date())
    df['hour'] = df.index.map(lambda x: x.hour)
    df['day_of_month'] = df.index.map(lambda x: x.day)
    # from 0 to 6
    df['day_of_week'] = df.index.map(lambda x: x.dayofweek)
    df['month'] = df.index.map(lambda x: x.month)
    df['year'] = df.index.map(lambda x: x.year)

    return df

train = preprocess(train)
train = train.drop_duplicates(['timestamp'])
old_train_length = len(train)

test = preprocess(test)
old_test_length = len(test)


train_dates = set(train['date'].unique())
test_dates = set(test['date'].unique())

combined = pd.concat([train, test], axis=0)
combined = combined.sort_index()

combined['isTest'] = combined['pm25'].isna()

# TODO: List of trains and tests for TES like for Rossman (see this code)
# in Rossman actually I had store id - maybe here I could have 'period_id' or sth and use like for rossman
# TODO: For XGB it will be about having proper aggregates in train and test df's

combined['period_id'] = 0
period_idx = 0
is_train = True
for i, (_, row) in enumerate(combined.iterrows()):
    if is_train and row['isTest']:
        is_train = False
    elif not is_train and not row['isTest']:
        is_train = True
        period_idx += 1
    combined['period_id'].iloc[i] = period_idx

# there should be 50 periods ideas because he said that data was created by 50 iterations
assert combined['period_id'].min() == 0 and combined['period_id'].max() == 49 and combined['period_id'].nunique() == 50

train = combined[combined['isTest'] == False]
assert len(train) == old_train_length
test = combined[combined['isTest']]
assert len(test) == old_test_length


train.to_csv('../data_processed/period_id_added/train.csv')

# leave only reasonable columns - timestamp, id and all dervied from timestamp
test = test[['id', 'pm25', 'date', 'hour', 'day_of_month', 'day_of_week', 'isTest', 'period_id', 'month', 'year']]
test.to_csv('../data_processed/period_id_added/test.csv')
print('done')
