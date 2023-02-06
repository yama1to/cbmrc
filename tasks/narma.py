# Copyright (c) 2022 Katori lab. All Rights Reserved
"""
NARMAタスク
"""
from generate_datasets.generate_data_sequence_narma import generate_narma
import numpy as np
import matplotlib.pyplot as plt

def dataset(MM1,MM2,delay=9,seed1=1,seed2=2):

    U1,D1= generate_narma(MM1,delay,seed1)
    U2,D2= generate_narma(MM2,delay,seed2)

    return U1,D1,U2,D2
    
def evaluate(Yp,Dp,):
    def calc(Yp,Dp):
        error = (Yp-Dp)**2
        NMSE = np.mean(error)/np.var(Dp)
        RMSE = np.sqrt(np.mean(error))
        NRMSE = RMSE/np.var(Dp)
        return RMSE,NRMSE,NMSE

    Yp = np.tanh(Yp)
    Dp = np.tanh(Dp)
    RMSE,NRMSE,NMSE = calc(Yp,Dp)
    #print(1/np.var(Dp))
    print('RMSE ={:.3g} NMSE ={:.3g} NRMSE ={:.3g} '.format(RMSE,NMSE,NRMSE))
    return RMSE,NMSE,NRMSE

def plot(c,U,X,Y,D,dir_name = "saved_figures",fig_name="mc1"):
    Nr=3
    
    plt.figure(figsize=(16, 10))

    plt.subplot(Nr,1,1)
    plt.plot(U)
    plt.xlabel("time steps")
    plt.ylabel("Input (U)")

    plt.subplot(Nr,1,2)
    plt.plot(X)
    plt.xlabel("time steps")
    plt.ylabel("Reservoir (X)")

    plt.subplot(Nr,1,3)
    plt.plot(Y)
    plt.plot(D)
    plt.xlabel("time steps")
    plt.ylabel("Output (Y,D)")
    
    if c.savefig:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if c.show :plt.show()
