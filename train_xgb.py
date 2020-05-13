import functools
import re

from helper import *
from predict_one_by_one import predict_one_by_one
from preprocessing.aggregates_preprocessing import add_rolling_aggregates_lag_one
from utils import read_data, save_submission, convert_to_float_or_factorize_objects
from xgb_custom_training import custom_valid_scheme

target = 'pm25'

# this is function for recalculating the rooling aggregates in one-by-one prediction
# it doesn't cover whole preprocessing
AGG_FUNCTION = functools.partial(add_rolling_aggregates_lag_one, rolling_lag=5)

SUBMISSION_NAME = 'forecasts_with_aggregates_lag_one_deviations_and_temperature_max_depth_at_9_gamma'
DATA_NAME = 'forecasts_with_aggregates_lag_one_deviations_and_temperature'
print(DATA_NAME)

model = xgb.XGBRegressor(objective='reg:squarederror')

xgb_params = { 'max_depth': 9, 'n_estimators': 5, 'learning_rate': 0.15, 'colsample_bytree': .7, 'subsample': 0.7, 'seed': 2018, 'gamma': 0.01}

train_and_valid, test = read_data('./data_processed/%s' % DATA_NAME, ext='h5')


basic_feats = ['hour', 'day_of_month', 'day_of_week', 'month', 'year', 'median_by_hour', 'deviation_from_previous']
rolling_mean_feats = [col for col in train_and_valid.columns if (('rolling' in col) and (int(re.search('[0-9]+', col).group(0)) in [1, 2, 3, 4, 5]) and ('median' not in col)) and ('hour' not in col)]
print(rolling_mean_feats)


forecast_feats = [
       'forecasted_totalSnow_cm', 'forecasted_sunHour', 'forecasted_uvIndex',
       'forecasted_uvIndex.1', 'forecasted_moon_illumination',
       'forecasted_moonrise', 'forecasted_moonset', 'forecasted_sunrise',
       'forecasted_sunset', 'forecasted_DewPointC', 'forecasted_FeelsLikeC',
       'forecasted_HeatIndexC', 'forecasted_WindChillC',
       'forecasted_WindGustKmph', 'forecasted_cloudcover',
       'forecasted_humidity', 'forecasted_precipMM', 'forecasted_pressure',
       'forecasted_tempC', 'forecasted_visibility', 'forecasted_winddirDegree',
       'forecasted_windspeedKmph']

print(forecast_feats)
feats = basic_feats + rolling_mean_feats + forecast_feats
print(feats)

train_and_valid = convert_to_float_or_factorize_objects(train_and_valid, feats)
train = train_and_valid[train_and_valid['split'] == 'train']
X_train = train[feats]
y_train = train[target]

valid = train_and_valid[train_and_valid['split'] == 'valid']
X_valid = valid[feats]
eval_set = [(X_valid, valid[target])]
model = XGBWrapper(xgb_params, early_stopping_rounds=30, eval_set=eval_set, verbose=10)

val_score = custom_valid_scheme(model, train, valid, feats, target, agg_function=AGG_FUNCTION)

test = convert_to_float_or_factorize_objects(test, feats)

test = predict_one_by_one(train=train_and_valid, test=test, feats=feats, model=model,
                                         agg_function=AGG_FUNCTION)
assert test.index.is_monotonic_increasing

save_submission(test, '%s.csv' % SUBMISSION_NAME, val_score=val_score)

model.plot_importance()

print('done')
