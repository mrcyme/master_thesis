import xgboost_surrogates as XGB_SUR
import kriging_surrogates as KRG_SUR
from samplers import *
from brock_hommes import BrockHommes
import json


def run_xgboost(budget,N_exp,train_sampler,test_sampler):
    X_domain_range= json.load(open("config.json",'r+'))
    X_train= train_sampler(X_domain_range,N_exp)
    y_train = np.empty(X_train.shape[0])
    for row,i in zip(X_train,range(X_train.shape[0])):
        A = BrockHommes(row)
        y_train[i]=A.simulate()

    X_test =test_sampler(X_domain_range,N_exp)
    y_test = np.empty(X_train.shape[0])
    for row,i in zip(X_test,range(X_test.shape[0])):
        A = BrockHommes(row)
        y_test[i]=A.simulate()
    surrogate = XGB_SUR.fit_model_regression(X_train,y_train)
    y_hat_test = surrogate.predict(X_test)



#run_xgboost(1000,10,sobol_sampling,random_sampling)


def run_kriging(N_exp,train_sampler,test_sampler):
    X_domain_range= json.load(open("config.json",'r+'))
    X_train= train_sampler(X_domain_range,N_exp)
    y_train = np.empty(X_train.shape[0])
    for row,i in zip(X_train,range(X_train.shape[0])):
        A = BrockHommes(row)
        y_train[i]=A.simulate()

    X_test =test_sampler(X_domain_range,N_exp)
    y_test = np.empty(X_train.shape[0])
    for row,i in zip(X_test,range(X_test.shape[0])):
        A = BrockHommes(row)
        y_test[i]=A.simulate()

    surrogate = KRG_SUR.fit_model_regression(X_train,y_train)

    y_hat_test = surrogate.predict(X_test)
    print(y_hat_test)
    print(y_test)

run_kriging(10,sobol_sampling,random_sampling)
