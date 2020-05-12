import pandas as pd
import numpy as np
df_sp_500 = pd.read_csv("^GSPC.csv")

df_sp_500 = df_sp_500.iloc[0:500]
def conpute_log_return(df):
    adj_close_price=df['Adj Close'].values
    log_return = np.diff(np.log(adj_close_price))
    pd.DataFrame(log_return,columns=['log_return']).to_csv("log_return.csv")


conpute_log_return(df_sp_500)
