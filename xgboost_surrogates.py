from sklearn.metrics import mean_squared_error, f1_score
from scipy.stats.distributions import entropy
import numba
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from skopt import gp_minimize
import numpy as np

_N_EVALS = 10
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00



def get_regression_error(y_hat, y):
    return mean_squared_error(y.get_label(), y_hat)
def get_classification_error(y_hat, y):
    return f1_score(y.get_label(), y_hat, average='weighted')


def xgboost_surrogates():
    surrogate_model = XGBRegressor(seed=0)
    surrogate_parameter_space = [
        (100, 1000),  # n_estimators
        (0.01, 1),  # learning_rate
        (10, 1000),  # max_depth
        (0.0, 1),  # reg_alpha
        (0.0, 1),  # reg_lambda
        (0.25, 1.0)]  # subsample
    return surrogate_model, surrogate_parameter_space



def fit_model_regression(X, y):
    surrogate_model, surrogate_parameter_space = xgboost_surrogates()
    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha,reg_lambda, subsample = params
        reg = XGBRegressor(n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           reg_alpha=reg_alpha,
                           reg_lambda=reg_lambda,
                           subsample=subsample,
                           seed=0)

        kf = KFold(n_splits=_N_SPLITS, random_state=0, shuffle=True)
        kf_cv = [(train, test) for train, test in kf.split(X, y)]
        return -np.mean(cross_val_score(reg,
                                        X, y,
                                        cv=kf_cv,
                                        n_jobs=1,
                                        fit_params={'eval_metric':get_regression_error},
                                        scoring="neg_mean_squared_error"))

    # use Gradient Boosted Regression to optimize the Hyper-Parameters.
    surrogate_model_tuned = gp_minimize(objective,
                                        surrogate_parameter_space,
                                        n_calls=_N_EVALS,
                                        acq_func='gp_hedge',
                                        n_jobs=-1,
                                        random_state=0, verbose=9)

    surrogate_model.set_params(n_estimators=surrogate_model_tuned.x[0],
                               learning_rate=surrogate_model_tuned.x[1],
                               max_depth=surrogate_model_tuned.x[2],
                               reg_alpha=surrogate_model_tuned.x[3],
                               reg_lambda=surrogate_model_tuned.x[4],
                               subsample=surrogate_model_tuned.x[5],
                               seed=0)

    surrogate_model.fit(X, y, eval_metric=get_regression_error)
    return surrogate_model



def get_round_selections(evaluated_set_X, evaluated_set_y,unevaluated_set_X,predicted_positives, num_predicted_positives,samples_to_select, calibration_threshold,budget):
    """
    """
    samples_to_select = np.min([abs(budget - evaluated_set_y.shape[0]),samples_to_select]).astype(int)

    if num_predicted_positives >= samples_to_select:
        round_selections = int(samples_to_select)
        selections = np.where(predicted_positives == True)[0]
        selections = np.random.permutation(selections)[:round_selections]

    elif num_predicted_positives <= samples_to_select:
        # select all predicted positives
        selections = np.where(predicted_positives == True)[0]
        # select remainder according to entropy weighting
        budget_shortfall = int(samples_to_select - num_predicted_positives)
        selections = np.append(selections,
                               get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X,
                                                      calibration_threshold,
                                                      budget_shortfall))

    else:  # if we don't have any predicted positive calibrations
        selections = get_new_labels_entropy(clf, unevaluated_set_X, samples_to_select)

    to_be_evaluated = unevaluated_set_X[selections]
    unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y, evaluate_islands_on_set(to_be_evaluated))

    return evaluated_set_X, evaluated_set_y, unevaluated_set_X


def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels):
    """ Get a set of parameter combinations according to their predicted label entropy
    """
    clf = fit_entropy_classifier(evaluated_set_X, evaluated_set_y, calibration_threshold)

    y_hat_probability = clf.predict_proba(unevaluated_X)
    y_hat_entropy = np.array(map(entropy, y_hat_probability))
    y_hat_entropy /= y_hat_entropy.sum()
    unevaluated_X_size = unevaluated_X.shape[0]

    selections = np.random.choice(a=unevaluated_X_size,
                                  size=number_of_new_labels,
                                  replace=False,
                                  p=y_hat_entropy)
    return selections
