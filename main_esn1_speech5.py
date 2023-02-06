# Copyright (c) 2022 Katori lab. All Rights Reserved
import argparse
import numpy as np
import models.esn1 as model
import tasks.speech5 as task
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
        self.project = "esn1_speech5"
        
        ### model:cbmrc
        self.seed:int=1     # 乱数生成のためのシード
        self.NS = 50      # 時間ステップ数（サイクル数） 
        self.NS0 = 0      # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）

        self.Nu = 77         # ノード数（入力）
        self.Nx = 200       # ノード数（レザバー）
        self.Ny = 10         # ノード数（出力）
        
        self.alpha_i = 0.5  # 結合強度（入力）
        self.alpha_r = 0.8  # 結合強度（レザバーのリカレント結合）
        self.alpha_b = 0.   # 結合強度（フィードバック）

        self.beta_i = 0.3   # 結合率（入力）
        self.beta_r = 0.9   # 結合率（レザバーのリカレント結合）
        self.beta_b = 0.    # 結合率（フィードバック）

        self.lambda0 = 0. # Ridge回帰
        
        ### Results
        self.WSR = None
        self.do_not_use_tqdm=True
        
def execute(c):
    Utrain,Ytrain,Utest,Ytest,shape = task.dataset(data=1,load=1,)
    (num_train,num_test,num_steps_cochlear,num_freq,Nclass) = shape
    model.c=c
    model.initialize() # 初期化

    Ypred,Ytrain = task.run_model(c,model,Utrain,Ytrain,num_train,num_steps_cochlear,mode="train")
    c.trainWSR = task.evaluate(Ypred,Ytrain,num_train,mode='train')

    Ypred,Ytest = task.run_model(c,model,Utest,Ytest,num_test,num_steps_cochlear,mode="test")
    c.WSR = task.evaluate(Ypred,Ytest,num_test,mode='test')

    if c.plot: task.plot(c,Utest[c.NS0:],model.X[c.NS0:],Ypred[c.NS0:],Ytest[c.NS0:])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
