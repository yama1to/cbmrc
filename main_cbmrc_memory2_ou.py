# Copyright (c) 2022 Katori lab. All Rights Reserved

import argparse
from fileinput import filename
import numpy as np
import sys
import os

from explorer import common 

from models.utils import load_model,save_model
import models.cbmrc as rc
import tasks.memory2_ou as task
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
        self.project = "cbmrc_memory2_ou"
        
        ### model:cbmrc
        self.seed:int=1 # 乱数生成のためのシード
        self.NN = 2**8 # １サイクルあたりの時間ステップ
        self.NS = 2200 # 時間ステップ数（サイクル数） 
        self.NS0 = 50 # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）

        self.Nu = 1     # ノード数（入力）
        self.Nh = 100   # ノード数（レザバー）
        self.Ny = 20     # ノード数（出力）

        self.Temp=1     # 温度
        
        self.alpha_i = 0.24 # 結合強度（入力）
        self.alpha_r = 0.1  # 結合強度（レザバーのリカレント結合）
        self.alpha_b = 0.   # 結合強度（フィードバック）
        self.alpha_s = 0.5  # 結合強度（参照クロック）

        self.beta_i = 0.3   # 結合率（入力）
        self.beta_r = 0.9   # 結合率（レザバーのリカレント結合）
        self.beta_b = 0.    # 結合率（フィードバック）

        self.lambda0 = 0.0 # Ridge回帰

        self.is_load_model = False
        self.is_save_model = False

        ### task:memory
        self.delay = 20

        # ResultsX
        self.RMSE1=None
        self.RMSE2=None
        self.MC = None
        self.cnt_overflow=None
   
def execute(c):
    c.seed = int(c.seed)
    np.random.seed(c.seed)

    U,Y = task.dataset(c.delay,T=c.NS)

    if not c.is_load_model:
        model = rc.CBMRC(c) # 初期化
        model.fit(input=U,output=Y) # 訓練
        if c.is_save_model: save_model(model=model,project=c.project)
    else:
        model = load_model("20221127_165432.pickle")

    Ypred = model.run(input=U,output=Y)

    Us,Rs,Hx,Hp,Yp,c.cnt_overflow = model.show_recode()

    _,c.MC = task.evaluate(Ypred,Y,c.delay)
    
    print("MC={:.2f}".format(c.MC))
    print("OverFlow={:.2f}".format(c.cnt_overflow))

    if c.plot:
        model.plot()
        task.plot(c,Ypred,Y,dir_name = "trashfigure",fig_name="mc1")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
