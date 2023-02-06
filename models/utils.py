# Copyright (c) 2022 Katori lab. All Rights Reserved
"""
Utilities for Reservoir Computing
"""
import numpy as np
import matplotlib.pyplot as plt

from copyreg import pickle
import datetime 
import pickle
import os


def prepare_directory(path):
    """
    # 指定されたディレクトリがなければ作る
    """
    if os.path.isdir(path): # ディレクトリの存在の確認
        #print("Exist Directory: %s" % (path))
        pass
    else:
        print("Create Directory: %s" % (path))
        os.makedirs(path) # ディレクトリの作成

SAVED_MODELS = "./saved_models/"
SAVED_FIGURES = "./saved_figures/"
prepare_directory(SAVED_MODELS)
prepare_directory(SAVED_FIGURES)

def string_now():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y%m%d_%H%M%S')
    return s

def save_model(model,fname=None,project=None):
    if fname == None:
        fname = string_now()+".pickle"
        
    if project != None:
        fname = string_now()+"_"+project+".pickle"

    with open(SAVED_MODELS+fname, mode='wb') as f:
        pickle.dump(model,f,protocol=2)
    
def load_model(filename):
    with open(SAVED_MODELS+filename, mode='rb') as f:
        model = pickle.load(f)
    return model

def file_name(fname):
    return os.path.basename(str(os.path.splitext(fname)[0]))

def plot2(Up,Hp,Yp,Dp,show = 1,save=1,dir_name = "trashfigure",fig_name="fig1"):
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
    if show :plt.show()
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))


def calc_MC(Yp,Dp,delay):
    DC = np.zeros(delay)
    for k in range(delay):
        corr = np.corrcoef(np.vstack((Dp.T[k, k:], Yp.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2    #決定係数 = 相関係数 **2
    MC = np.sum(DC)
    return DC,MC

def plot_MC(Yp,Dp,delay=20,show = 1,save=1,dir_name = "trashfigure",fig_name="mc1"):               
    DC,MC = calc_MC(Yp,Dp,delay)
    plt.plot(DC)
    plt.ylabel("determinant coefficient")
    plt.xlabel("Delay k")
    plt.ylim([0,1])
    plt.xlim([0,delay])
    plt.title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if show :plt.show()
    
def plot_delay(DC,N,pred,target,):
    fig=plt.figure(figsize=(8,8 ))
    Nr=N
    start = 0
    for i in range(Nr):
        ax = fig.add_subplot(Nr,1,i+1)
        ax.cla()
        ax.set_title("DC = %2f,delay = %s" % (DC[i],str(i)))
        ax.plot(pred.T[i,i:])
        ax.plot(target.T[i,i:])
    plt.show()