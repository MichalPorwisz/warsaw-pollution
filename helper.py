import time

import matplotlib.pyplot as plt
import shap
import xgboost as xgb


class XGBWrapper:
    def __init__(self, xgb_params={}, early_stopping_rounds=10, eval_set=None, verbose=False):
        self.eval_metric = 'rmse'
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        self.verbose = verbose

        self.model = xgb.XGBRegressor(**xgb_params)
        self.X = None

    def fit(self, X, y):

        self.X = X
        if self.eval_set is None:
            self.eval_set = [(X, y)]

        self.model.fit(X, y, verbose=self.verbose, eval_metric=self.eval_metric, \
            eval_set=self.eval_set, early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, X):
        return self.model.predict(X)

    def plot_importance(self):
        time_str = f'{time.localtime(time.time()).tm_mday}_{time.localtime(time.time()).tm_hour}_{time.localtime(time.time()).tm_min}'

        for type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            fig, ax = plt.subplots(figsize=(20, 7))
            xgb.plot_importance(self.model.get_booster(), importance_type=type, ax=ax, max_num_features=20)
            plt.savefig(f'./visualizations/importances_{type}_{time_str}.png')
            plt.close()

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X, plot_type="bar", show=False, plot_size=(25, 7))
        plt.savefig(f'./visualizations/importances_shap_{time_str}.png')
        plt.close()

    def get_model(self):
        return self.model
