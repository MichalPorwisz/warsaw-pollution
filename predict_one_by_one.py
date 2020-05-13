import pandas as pd

from utils import convert_to_float_or_factorize_objects, flatten


def predict_one_by_one(train, test, feats, model, agg_function, target='pm25',
                       extract_test_func=lambda df: df[df['isTest']]):

    def predict_one_by_one_internal(combined, test_hour_ordinal, largest_lag=10):
        test = extract_test_func(combined)
        # here we have the rows for which we want to calculate aggregates and make predictions
        test_part = test[test['ordinal'] == test_hour_ordinal]

        # should works as long as largest_lag is really largest lag
        idxs_list = [list(range(idx - largest_lag, idx + 1)) for idx in test_part.index]
        idxs = flatten(idxs_list)
        df = agg_function(combined.iloc[idxs], 1)

        test_for_predictions = df.loc[test_part.index][feats]
        test_for_predictions = convert_to_float_or_factorize_objects(test_for_predictions, feats)
        pred = model.predict(test_for_predictions)
        test.loc[test_part.index, target] = pred

        return test

    new_test = test
    combined = pd.concat([train, new_test]).sort_index()
    combined = combined.reset_index(drop=False)
    for ordinal in range(24):
        new_test = predict_one_by_one_internal(combined, ordinal)
        new_test.index = new_test['timestamp']
        new_test = new_test.drop('timestamp', axis=1)
        combined = pd.concat([train, new_test]).sort_index()
        combined = combined.reset_index(drop=False)

    new_test = extract_test_func(combined)
    new_test.index = new_test['timestamp']
    new_test = new_test.drop('timestamp', axis=1)

    return new_test
