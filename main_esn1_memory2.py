# Copyright (c) 2022 Katori lab. All Rights Reserved
import argparse
import numpy as np
import models.esn1 as model
import tasks.memory2 as task
from explorer import common

class Config():
    # TODO pythonのdataclassに移行する.(Anaconda版pythonにdataclassが実装されてから) 
    def __init__(self):
        #self.columns = None 
        #self.csv = None 
        self.id  = None
        self.plot = True        # 図の出力
        self.show = True        # 図の表示
        self.savefig = False    # 図の保存
        self.fig1 = "fig1.png"  # 画像ファイル名
        self.project = "esn1_memory2"
        
        ### model:cbmrc
        self.seed:int=1     # 乱数生成のためのシード
        self.NS = 2000      # 時間ステップ数（サイクル数） 
        self.NS0 = 200      # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）

        self.Nu = 1         # ノード数（入力）
        self.Nx = 20       # ノード数（レザバー）
        self.Ny = 20         # ノード数（出力）
        
        self.alpha_i = 0.1  # 結合強度（入力）
        self.alpha_r = 0.8  # 結合強度（レザバーのリカレント結合）
        self.alpha_b = 0.   # 結合強度（フィードバック）

        self.beta_i = 0.1   # 結合率（入力）
        self.beta_r = 0.1   # 結合率（レザバーのリカレント結合）
        self.beta_b = 0.    # 結合率（フィードバック）

        self.lambda0 = 0. # Ridge回帰

        ### task:narma
        self.delay = 20
        
        ### Results
        self.MC = None
        
def execute(c):
    Utrain,Ytrain = task.dataset(delay=c.delay,T=c.NS,dist='uniform')
    Utest,Ytest = Utrain,Ytrain
    model.c=c
    model.initialize() # 初期化
    model.fit(input=Utrain,output=Ytrain) # 訓練
    Ypred = model.run(input=Utest) # テスト
    c.DC ,c.MC = task.evaluate(Ypred[c.NS0:],Ytest[c.NS0:],c.delay) # 評価
    if c.plot: task.plot(c,Ypred[c.NS0:],Ytest[c.NS0:], dir_name= "trashfigure", fig_name= "mc1")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
