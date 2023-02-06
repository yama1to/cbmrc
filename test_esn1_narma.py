# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""

"""
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

### 共通設定
from main_esn1_narma import Config
config = Config()
common.config  = config
common.prefix  = "data%s_esn1_narma" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_esn1_narma" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python main_esn1_narma.py " # 実行されるプログラム
common.columns =['seed','id','Nx','alpha_i','alpha_r','alpha_b','beta_i','beta_r','beta_b','lambda0','delay',"RMSE",'NRMSE',"NMSE"]
common.parallel= 120
common.setup()
common.report_common()
common.report_config(config)

### ランダムサーチ
def rs1():
    rs.clear()
    rs.append("alpha_r",min=0,max=5)
    rs.append("alpha_i",min=0,max=5)
    rs.random(num=60,samples=2)
    df = common.load_dataframe() # 直前に保存されたcsvファイルをデータフレーム(df)に読み込む
    df = df[['alpha_r','alpha_i','cnt_overflow','NMSE']] # 指定した列のみでデータフレームを構成する
    #df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    # 変数の追加([変数名],[基本値],[下端],[上端],[まるめの桁数])

    opt.append("beta_r",value=0.01,min=0.,max=1,round=2)
    opt.append("beta_i",value=0.01,min=0.,max=1,round=2)
    opt.append("alpha_i",value=0.9,min=0.,max=1,round=2)
    opt.append("alpha_r",value=1,min=0.,max=1,round=2)
    opt.append("alpha_s",value=1,min=0,max=2,round=2)
    #opt.append("Temp",value=1,min=1,max=10,round=2)
    opt.minimize(target="NMSE",iteration=30,population=30,samples=4)
    #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
#optimize()

def plot1(x,y,ystd,ymin,ymax,color=None,width=1,label=None):
    # エラーバーをつけてグラフを描画、平均、標準偏差、最大値、最小値をプロットする。
    #ax.errorbar(x,y,yerr=ystd,fmt='o',color=color,capsize=2,label="xxxx")
    plt.plot(x,y,color=color,linestyle='-',linewidth=width,label=label)
    plt.fill_between(x,y-ystd,y+ystd,color=color,alpha=.2)
    plt.plot(x,ymin,color=color,linestyle=':',linewidth=1)
    plt.plot(x,ymax,color=color,linestyle=':',linewidth=1)

def gridsearch(X1,min=0,max=1,num=41,samples=10):
    # 指定された変数(X1)についてグリッドサーチを行い、評価基準の変化をまとめてプロット

    gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)
    df = common.load_dataframe()
    #print(df)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,8))

    plt.subplot(2,1,1)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"NMSE")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="NMSE")
    plt.ylabel("NMSE")
    plt.yscale('log')
    plt.grid(linestyle="dotted")
    #plt.ylim([0,1]) # y軸の範囲

    plt.xlabel(X1)
    vs.plt_output()

def gs2():
    ns=4
    #gridsearch("Nh",min=100,max=1000,num=41,samples=ns)
    #gridsearch("delay",min=1,max=9,num=10,samples=ns)
    gridsearch("alpha_i",min=0.,max=1,num=41,samples=ns)
    gridsearch("alpha_r",min=0.,max=1.5,num=41,samples=ns)
    #gridsearch("beta_i",min=0.0,max=1,num=41,samples=ns)
    #gridsearch("beta_r",min=0.0,max=1,num=41,samples=ns)
    # gridsearch("Temp",min=0.0,max=10,num=41,samples=ns)
    #gridsearch("lambda0",min=0.0,max=10,num=41,samples=ns)
gs2()
