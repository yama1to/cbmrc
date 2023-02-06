from generate_datasets.generate_data_sequence_ipc2 import datasets
import numpy as np
import matplotlib.pyplot as plt
from explorer import common


global name_list,dist_list
name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
dist_list = ["uniform","normal","arcsine","exponential"]

def dataset(MM,set=0,delay=20,degree=10,seed=0):
    
    dist = dist_list[set]
    name = name_list[set]

    U,D = datasets(k=delay,n=1,T = MM,name=name,dist=dist,seed=seed,new=0)
    for deg in range(2,degree+1):
        _,D2 = datasets(k=delay,n=deg,T = MM,name=name,dist=dist,seed=seed,new=0)
        D = np.hstack([D,D2])

    return U,D



def evaluate(Yp,Dp,delay=20):
    MC = 0
    CAPACITY = []

    # print(Dp.shape,Yp.shape)
    for i in range(delay*10):
        r = np.corrcoef(Dp[delay:,i],Yp[delay:,i])[0,1]
        CAPACITY.append(r**2)
    MC = sum(CAPACITY)
    # ep = 1.7*10**(-4)
    # MC = np.heaviside(MC-ep,1)*MC
    
    ERR = np.sum((Yp-Dp)**2)

    return MC,CAPACITY,ERR


def plot(c,Up,Hp,Yp,Dp,dir_name = "trashfigure",fig_name="mc1"):
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    #ax.set_title("input")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    #ax.set_title("decoded reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    #ax.set_title("predictive output")
    #ax.plot(train_Y)
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    
    if c.savefig:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if c.show   :plt.show()

def plot2(c):
    c.dist = dist_list[c.set]
    c.name = name_list[c.set]
    t = common.string_now()

    for i in range(10):
        plt.plot(c.CAPACITY[i*(20):(i+1)*20],label="degree = "+str(1+i))

    plt.ylabel("Capacity")
    plt.xlabel("delay")
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,20.1])
    plt.title("cbm::"+c.name+"::"+c.dist)
    plt.legend()
    
    if c.savefig:
        na = "./saved_figures/ipc-fig/%s-ipc4_cbm_fixed_in_tar_%s.eps" % (t,str(c.set))
        plt.savefig(na)
        na = "./saved_figures/ipc-fig/%s-ipc4_cbm_fixed_in_tar_%s.png" % (t,str(c.set))
        plt.savefig(na)
    if c.show :plt.show()
    plt.clf()
    