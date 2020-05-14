import numpy as np
from ml_metrics import rmse

from predict_one_by_one import predict_one_by_one


def custom_valid_scheme(model, train, valid, feats, target, agg_function, early_stopping=5, val_at_num_epoch=5):

    def _train(X_train, model, y_train, iteration):
        # TODO: Train based on model from previous iteration instead of from scratch (although it's not a real bottleneck)
        model.get_model().set_params(n_estimators=(iteration * val_at_num_epoch))
        model.fit(X_train, y_train)

    extract_test_func = lambda df: df[df['split'] == 'valid']

    X_train = train[feats]
    y_train = train[target]

    epochs_without_improvement = 0
    best_score = np.inf
    best_iter = 0
    iter = 1
    while epochs_without_improvement < early_stopping:
        _train(X_train, model, y_train, iter)

        new_valid = predict_one_by_one(train=train, test=valid, feats=feats, model=model, agg_function=agg_function,
                                       extract_test_func=extract_test_func)
        score = rmse(valid[target].values, new_valid[target].values)
        print(f'RMSE on valid: {score}')

        if score < best_score:
            best_score = score
            best_iter = iter
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        iter += 1

    model.get_model().set_params(n_estimators=(best_iter * val_at_num_epoch))
    model.fit(X_train, y_train)

    print(f'score didn\'t improve for {epochs_without_improvement} epochs - finished training with best score of {best_score}')
    return best_score

