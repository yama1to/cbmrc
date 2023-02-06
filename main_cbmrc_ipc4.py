# Copyright (c) 2022 Katori lab. All Rights Reserved

import argparse
from fileinput import filename
import numpy as np
import sys
import os

from explorer import common

from models.utils import load_model,save_model
import models.cbmrc as rc
import tasks.ipc as task
from explorer import common

class Config():
    def __init__(self):
        #self.columns = None # 
        #self.csv = None # 
        self.id  = None
        self.plot = True        # 図の出力
        self.show = True        # 図の表示
        self.savefig = False    # 図の保存
        self.fig1 = "fig1.png"  # 画像ファイル名
        self.project = "cbmrc_ipc4"

        # config
        self.dataset=6
        self.seed:int=1 # 乱数生成のためのシード
        self.NN=2**8 # １サイクルあたりの時間ステップ
        self.NS=2200 # サイクル数
        self.NS0 = 200 #

        self.Nu = 1         #size of input
        self.Nh:int = 50   #815 #size of dynamical reservior
        self.Ny = 200        #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.12
        self.alpha_r = 0.3
        self.alpha_b = 0.
        self.alpha_s = 0.5

        self.alpha0 = 0#0.1
        self.alpha1 = 1#-5.8

        self.beta_i = 0.1
        self.beta_r = 0.1
        self.beta_b = 0.

        self.lambda0 = 0

        self.delay = 20
        self.degree = 10
        self.set = 0    #0,1,2,3

        self.is_load_model = False
        self.is_save_model = False

        # ResultsX
        self.cnt_overflow=None

        self.MC = None

def execute(c):
    c.seed = int(c.seed)
    np.random.seed(c.seed)

    U,Y = task.dataset(c.NS,set=c.set,delay=c.delay,degree=c.degree,seed=c.seed)

    if not c.is_load_model:
        model = rc.CBMRC(c) # 初期化 
        model.fit(input=U,output=Y) # 訓練
        if c.is_save_model: save_model(model=model,project=c.project)
    else:
        model = load_model("20221127_165432.pickle")

    Ypred = model.run(input=U,output=Y) # テスト

    Us,Rs,Hx,Hp,Yp,c.cnt_overflow = model.show_recode()

    c.MC, c.CAPACITY,c.ERR = task.evaluate(Ypred,Y)
    print(c.MC,c.ERR)
    print("OverFlow={:.2f}".format(c.cnt_overflow))

    if c.plot:
        model.plot()
        task.plot(c,U,Hp,Yp,Y,dir_name = "trashfigure",fig_name="mc1")
        task.plot2(c)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
