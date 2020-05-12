import pandas as pd
import numpy as np
import json
from scipy.stats import ks_2samp
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error, f1_score






class BrockHommes:
    def __init__(self,x):
        self.g1=x[0]
        self.g2=x[1]
        self.b1=x[2]
        self.b2=x[3]
        self.alpha=x[4]
        self.sigma=x[5]
        self.R=x[6]
        self.w=x[7]
        self.beta=x[8]
        self.C=x[9]
        self.N=500
        self.price=np.concatenate((np.array([1,self.R*1]),np.empty((self.N))))
        self.n=np.concatenate((np.array([0.5,0.5]),np.empty((self.N))))
        self.U1=np.concatenate((np.array([0]),np.empty((self.N))))
        self.U2=np.concatenate((np.array([0]),np.empty((self.N))))



    def update_price(self,t):
        self.price[t]= (self.n[t-1]*(self.g1*self.price[t-1]+self.b1)+(1-self.n[t-1])*(self.g2*self.price[t-1]+self.b2)+(np.random.uniform(1)/1000))/self.R

    def update_accumulated_profit(self,t):
        self.U1[t-1] = (1/(self.alpha*self.sigma**2))*(self.price[t]-self.R*self.price[t-1])*(self.g1*self.price[t-2] + self.b1 - self.R*self.price[t-1]) +self.w*self.U1[t-2] -self.C
        self.U2[t-1] = (1/(self.alpha*self.sigma**2))*(self.price[t]-self.R*self.price[t-1])*(self.g2*self.price[t-2] + self.b2 - self.R*self.price[t-1]) +self.w*self.U2[t-2] -self.C

    def update_fraction(self,t):
        self.n[t] = np.exp(self.beta*self.U1[t-1])/(np.exp(self.beta*self.U1[t-1])+np.exp(self.beta*self.U2[t-1]))

    def compute_log_return(self):
        log_return = np.diff(np.log(self.price))
        return log_return

    def compute_calibration_metric(self):
        r_emp = pd.read_csv("data/log_return.csv")['log_return'].values
        r_sim = self.compute_log_return()
        p_val = ks_2samp(r_sim,r_emp)[1]
        return p_val


    def simulate(self):
        for t in range(2,self.N+2):
            self.update_price(t)
            self.update_accumulated_profit(t)
            self.update_fraction(t)
        self.price=self.price[2:]
        self.n=self.n[2:]
        self.U1=self.U1[1:]
        self.U2=self.U2[1:]
        return self.compute_calibration_metric()



"""
def run_exp(N):
    config = json.load(open("config.json",'r+'))
    df = sobol_sampling(config,N)
    res = np.empty((N))
    for index, row in df.iterrows():
        sim = Simulation(row.to_dict())
        r_sim = sim.simulate()
        p_val = ks_2samp(r_sim,r_emp)[1]
        res[index]=p_val
    #print(res)
    df['p_value']=res
    df['label']= df['p_value']>0.05
    df.to_csv("data.csv")


data = pd.read_csv("data.csv")
"""
