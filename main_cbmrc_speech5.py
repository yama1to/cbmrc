# Copyright (c) 2022 Katori lab. All Rights Reserved
import argparse
import numpy as np
import sys
import os
from explorer import common

from models.utils import load_model,save_model
import models.cbmrc as rc
import tasks.speech5 as task
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
        self.project = "cbmrc_speech5"
        
        ### model:cbmrc
        self.seed:int=1 # 乱数生成のためのシード
        self.NN = 2**8 # １サイクルあたりの時間ステップ
        self.NS = 50 # 時間ステップ数（サイクル数） 
        self.NS0 = 0 # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）

        self.Nu = 77     # ノード数（入力）
        self.Nh = 300   # ノード数（レザバー）
        self.Ny = 10     # ノード数（出力）

        self.Temp=1     # 温度
        
        self.alpha_i = 0.5 # 結合強度（入力）
        self.alpha_r = 0.02  # 結合強度（レザバーのリカレント結合）
        self.alpha_b = 0.   # 結合強度（フィードバック）
        self.alpha_s = 0.4  # 結合強度（参照クロック）

        self.beta_i = 0.05   # 結合率（入力）
        self.beta_r = 0.02   # 結合率（レザバーのリカレント結合）
        self.beta_b = 0.    # 結合率（フィードバック）

        self.lambda0 = 0.0 # Ridge回帰

        self.is_load_model = False
        self.is_save_model = False
        self.do_not_use_tqdm = True

        self.trainWSR = None
        self.WSR = None
        self.cnt_overflow=None


def execute(c):
    np.random.seed(int(c.seed))
    Utrain,Ytrain,Utest,Ytest,shape = task.dataset(data=1,load=1)
    (num_train,num_test,num_steps_cochlear,num_freq,Nclass) = shape

    print(Utrain.shape,Ytrain.shape)
    if not c.is_load_model:
        model = rc.CBMRC(c)
        y,d = task.run_model(c,model,Utrain,Ytrain,num_train,num_steps_cochlear,mode="train")
        c.trainWSR = task.evaluate(y,d,num_train,mode='train')
        if c.is_save_model: save_model(model=model,project=c.project)
    else:
        model = load_model('20220321_071446_'+__file__+'.pickle')

    y,d = task.run_model(c,model,Utest,Ytest,num_test,num_steps_cochlear,mode='test')
    c.WSR = task.evaluate(y,d,num_test,mode='test')

    Us,Rs,Hx,Hp,Yp,c.cnt_overflow = model.show_recode()
    
    print("OverFlow={:.2f}".format(c.cnt_overflow))

    if c.plot: 
        model.plot()
        task.plot(c,Utest,Hp,Yp,Ytest,dir_name = "trashfigure",fig_name="mc1")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()
    
    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
