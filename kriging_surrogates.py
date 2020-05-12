import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,ExpSineSquared, DotProduct,ConstantKernel)
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern,RBF
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from skopt import gp_minimize


_N_EVALS = 10
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00


def get_regression_error(y_hat, y):
    return mean_squared_error(y.get_label(), y_hat)
def get_classification_error(y_hat, y):
    return f1_score(y.get_label(), y_hat, average='weighted')



def kriging(X,y,kernel):
    krg = GaussianProcessRegressor(kernel=kernel).fit(X, y)
    return krg


def kriging_surrogates():
    surrogate_model = GaussianProcessRegressor(normalize_y=True)
    kernels = [1.0 * Matern(nu=5/2),1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))]
    params = [kernels]
    return surrogate_model, params

def fit_model_regression(X, y):

    surrogate_model, params = kriging_surrogates()
    def objective(kernel):
        reg = GaussianProcessRegressor(normalize_y=True,kernel=kernel,seed=0)
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
                                        params,
                                        n_calls=_N_EVALS,
                                        acq_func='gp_hedge',
                                        n_jobs=-1,
                                        random_state=0, verbose=9)

    surrogate_model.set_params(kernel=surrogate_model_tuned.x[0],seed=0)
    surrogate_model.fit(X, y, eval_metric=get_regression_error)
    return surrogate_model
