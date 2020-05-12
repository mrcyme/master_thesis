from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer
from pdb import set_trace as keyboard


#NOT YET


class DNNSurrogate(object):
    surrogate_parameter_space = [
        (30, 100),  # L : first hidden layer
        (1,30), #d last hidden layer of data compression
        (0, 0.0002),  # learning_rate
        (0, 0.000002), #regularisation
        ]
    def fit(self, input_dim,params):
        """
        X, Y -> Input output data.
        D -> Input dimensionality.
        L -> Number of layers in encoding section.
        d -> Size of the encoding layer.
        baselr -> Base learning rate (default: 1e-3)
        lmbda -> Reg. constant. (default: 1e-6)
        """
        self.L = params[0]
        self.d = params[1]
        self.rho = (1./self.L)*np.log(self.d/(self.D*1.))
        self.ds = [self.D] + [int(np.ceil(self.D*np.exp(self.rho*i))) for i in xrange(1, self.L+1)] +\[300*self.d, 1]
        self.baselr = param[2]
        self.lmbda = param[3]
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.define_params()
        self.define_graph()
        self.define_session()


    def define_params(self):
        xavinit = xavier_initializer()
        self.Ws = []
        self.bs = []
        with tf.variable_scope('parameters'):
            for i in xrange(1, len(self.ds)):
                hin = self.ds[i-1]
                hout = self.ds[i]
                W = tf.Variable(name='W_'+str(i), dtype=tf.float32, initial_value=xavinit(shape=[hin, hout]))
                b = tf.Variable(name='b_'+str(i), dtype=tf.float32, initial_value=xavinit(shape=[hout]))
                self.Ws.append(W)
                self.bs.append(b)

    def define_graph(self):
        with tf.variable_scope('activations'):
            self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.D])
            h = self.x
            self.tensors = [h]
            for i in xrange(len(self.Ws)):
                W = self.Ws[i]
                b = self.bs[i]
                h = tf.matmul(h, W) + b
                if b.shape[0] == self.d:
                    h = tf.identity(h, name='h_'+str(i+1))
                elif b.shape[0] == 1:
                    h = tf.identity(h, name='y')
                    self.y = h
                else:
                    h = tf.nn.relu(h, name='h_'+str(i+1))
                self.tensors.append(h)
        self.ytrue = tf.placeholder(name='ytrue', dtype=tf.float32, shape=[None, 1])

        with tf.variable_scope('optimizer'):
            self.mse = tf.reduce_mean(tf.squared_difference(self.y, self.ytrue))
            self.l1reg = l1_regularizer(self.lmbda)
            self.l2reg = l2_regularizer(self.lmbda)
            self.regparams = tf.add_n([self.l1reg(W)+self.l2reg(W) for W in self.Ws])
            self.loss = self.mse + self.regparams
            self.opt = tf.train.AdamOptimizer(self.baselr)
            self.step = self.opt.minimize(self.loss)

    def fit(X,y):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, x):
        return self.y.eval(feed_dict={self.x:x})

def fit_model_regression(X, y):

    surrogate_model, params = DNN_surrogates()
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
