import pandas as pd

REPLACE = True
input_path = '../data_processed/period_id_added/train.csv'
train = pd.read_csv(input_path, index_col=0)

train['split'] = 'train'

# TODO: last 24 points to validation set
prev = None
valid_points_per_period = 24
last = len(train) - 1
for i, (_, current) in enumerate(train.iterrows()):
    if prev is not None:
        current_period_id = current['period_id']
        prev_period_id = prev['period_id']
    if (prev is not None and current_period_id != prev_period_id):
        assert prev_period_id + 1 == current_period_id
        train['split'].iloc[i - valid_points_per_period: i] = 'valid'
    if i == last:
        train['split'].iloc[i - valid_points_per_period + 1: i + 1] = 'valid'

    prev = current

assert len(train[train['split'] == 'valid']) == train['period_id'].nunique() * valid_points_per_period
assert train['split'].iloc[last] == 'valid'

if REPLACE:
    train.to_csv(input_path)
else:
    # TODO: save with suffix
    raise Exception('not implemented')

