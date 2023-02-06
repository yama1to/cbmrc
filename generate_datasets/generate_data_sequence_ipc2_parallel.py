from re import X
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as st
from scipy.special import factorial
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import time
from explorer import common

# http://solidstatephysics.blog.fc2.com/blog-entry-44.html.
def Legendre(x,degree):
    P = 0
    for i in range(int(np.floor(degree/2)+1)):
        tmp = (-1)**i * factorial(2*degree-2*i) * x**(degree-2*i)
        tmp /= 2**degree * factorial(degree-i) * factorial(i) * factorial(degree-2*i)
        P += tmp
    return P

#http://solidstatephysics.blog.fc2.com/blog-entry-29.html.
def Hermite(x,degree):
    P = 0
    
    for k in range(int(np.floor(degree/2)+1)):
        tmp = (2*x)**(degree-2*k) * (-1)**k 
        tmp /= factorial(degree-2*k)*factorial(k)
        P += tmp
    P *=factorial(degree)
    return P

# http://math-juken.com/kijutu/chebyshev/
def Chebyshev(x,degree):
    T = 0
    for k in range(1+int(np.floor(degree/2))):
        t = (2*x)**(degree-2*k)
        tmp = (-1)**k*degree*factorial(degree-k)*t
        tmp /= 2*(degree-k)*factorial(degree-2*k)*factorial(k)
        T+= tmp
    return T

#https://www.ne.jp/asahi/music/marinkyo/matematiko/polinomoj-de-laguerre.html.ja
def Laguerre(x,degree):
    P = 0
    for k in range(degree+1):
        tmp = (-1)**k * factorial(degree)**2 * x**k
        tmp /= factorial(degree-k) * factorial(k)**2
        P += tmp
    return P

def polynomial(name,x,degree):
    if degree == 0: return np.ones(x.shape)
    if degree == 1: return x
    if name == "Legendre":
        return Legendre(x,degree)
    elif name == "Hermite":
        return Hermite(x,degree)
    elif name == "Chebyshev":
        return Chebyshev(x,degree)
    elif name == "Laguerre":
        return Laguerre(x,degree)

def datasets(k=2,n = 2,
                  T=1000,
                  name="Legendre",
                  dist="uniform",
                  #dist="exponential",
                  seed=0,
                  new=0,
                  save = 0,):
    """
    k:遅延長
    n:多項式の次数
    name:使用する多項式
    dist:入力の分布
    seed:乱数のシード値
    """
    DIR = "./generate_datasets/ipc_dir/"
    DIR += str(T)
    common.create_directory(DIR)
    DIR += "/"

    if not new:
        u = np.load(DIR+"input_"+str(T)+name+".npy")
        d = np.load(DIR+"target_"+str(T)+"_"+str(int(n))+name+".npy")
        return u,d
    np.random.seed(seed)

    if dist=="normal":
        u = np.random.normal(loc = 0,scale=0.3,size=(T,1))
    elif dist=="uniform":
        u = np.random.uniform(-1,1,(T,1))
    elif dist=="arcsine":
        u = st.arcsine.rvs(loc = -1,scale=2,size=(T,1))#-0.5
    elif dist=="exponential":
        u = st.expon.rvs(loc = -1,scale=0.22,size=(T,1))
    max = np.max(np.max(abs(u)))
    u = u/max/2


    delay = np.arange(k)  # 遅延長z 
    d = np.empty((T, len(delay)))
    
    # for t in range(T):
    #     for k in range(len(delay)):
    #         y = polynomial(name,u,n)

    #         d[t, k] = y[t-delay[k],0]  # 遅延系列
    # d = np.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            y = polynomial(name,u,n)
            d[t, k] = y[t-delay[k],0]  # 遅延系列

    if save:
                                #degree,delay,poly
        np.save(DIR+"input_"+str(T)+name+".npy",arr=u)
        np.save(DIR+"target_"+str(T)+"_"+str(int(n))+name+".npy",arr=d)
    return u,d

def for_func(pol,deg):
    i=pol
    datasets(k=20,n = deg,
                        T=1200, 
                        name=name_list[i],
                        dist=dist_list[i],
                        #dist="exponential",
                        seed=0,
                        new=1,
                        save = 1,)

def for_func1(idx,deg):
    poly = list(range(4))
    # tqdmで経過が知りたい時
    with tqdm(total=len(poly)) as progress:
        # 1. 引数にiterできないオブジェクトがある時
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
            futures = []  # 処理結果を保存するlist
            for i, pol in enumerate(poly):  # -------(3)
                future = executor.submit(for_func,pol,deg)
                future.add_done_callback(lambda p: progress.update()) # tqdmで経過が知りたい時
                futures.append(future)
            result = [f.result() for f in futures]

    return idx,deg


def for_func2():
    DEG = list(range(1,11))
    DEG = list(reversed(DEG))
    # tqdmで経過が知りたい時
    with tqdm(total=len(DEG)) as progress:
        # 1. 引数にiterできないオブジェクトがある時
        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # -----(2)
            futures = []  # 処理結果を保存するlist
            for i, deg in enumerate(DEG):  # -------(3)
                future = executor.submit(for_func1, i, deg)
                future.add_done_callback(lambda p: progress.update()) # tqdmで経過が知りたい時
                futures.append(future)
            result = [f.result() for f in futures]


if __name__=="__main__":
    global name_list, dist_list
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    

    for_func2()#10x4の並列化
          