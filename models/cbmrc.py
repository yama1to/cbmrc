# Copyright (c) 2022 Katori lab. All Rights Reserved
"""
Chaotic Boltzmann Machine based Reservoir Computing 
"""
import numpy as np
import matplotlib.pyplot as plt
from models.generate_matrix import *
from tqdm import tqdm

class CBMRC():
    """
    Reservoir Computing based on Chaotic Boltzmann Machine
    """
    def __init__(self,c):
        self.initialize(c)
        self.generate_network()
        self.cnt_overflow=None

    def initialize(self,c):
        self.c = c
        self.show = c.show      # 図の表示
        self.savefig = c.savefig# 図の保存
        self.fig1 = c.fig1      # 画像ファイル名

        self.seed:int = int(c.seed)  # 乱数生成のためのシード

        self.NN = c.NN          # １サイクルあたりの時間ステップ数 
        self.NS = c.NS          # 外部時間のステップ数（サイクル数） 
        self.NS0 = c.NS0        # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）
        self.NT = self.NS * self.NN # 内部時間のステップ数 
        self.dt = 1.0/self.NN   #　

        self.Nu = c.Nu          # ノード数（入力）
        self.Nh = c.Nh          # ノード数（レザバー）
        self.Ny = c.Ny          # ノード数（出力）

        self.Temp = c.Temp      # 温度
        
        self.alpha_i = c.alpha_i# 結合強度（入力）
        self.alpha_r = c.alpha_r# 結合強度（レザバーのリカレント結合）
        self.alpha_b = c.alpha_b# 結合強度（フィードバック）
        self.alpha_s = c.alpha_s# 結合強度（参照クロック）

        self.beta_i = c.beta_i  # 結合率（入力）
        self.beta_r = c.beta_r  # 結合率（レザバーのリカレント結合）
        self.beta_b = c.beta_b  # 結合率（フィードバック）

        self.lambda0 = c.lambda0 # Ridge回帰
        return None
    
    def generate_network(self,):
        np.random.seed(seed=self.seed)
        self.Wr = generate_random_matrix(self.Nh,self.Nh,self.alpha_r,self.beta_r,distribution="one",normalization="sr",diagnal=0)
        self.Wb = generate_random_matrix(self.Nh,self.Ny,self.alpha_b,self.beta_b,distribution="one",normalization="none")
        self.Wi = generate_random_matrix(self.Nh,self.Nu,self.alpha_i,self.beta_i,distribution="one",normalization="none")
        self.Wo = np.zeros(self.Nh * self.Ny).reshape((self.Ny, self.Nh))

    def fit(self,input,output):
        assert len(input)==len(output)
        self.run(input)
        self.regression(self.Hp,output)
        
    def regression(self,reservoir_state,output):
        M = reservoir_state[self.NS0:, :]
        G = output[self.NS0:, :]

        ### Ridge regression
        if self.lambda0 == 0:
            self.Wo = np.dot(G.T,np.linalg.pinv(M).T)
        else:
            E = np.identity(self.Nh)
            TMP1 = np.linalg.inv(M.T@M + self.lambda0 * E)
            WoT = TMP1@M.T@G
            self.Wo =WoT.T

    def p2s(self,theta,p):
        return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

    def run(self,input,output=np.empty(0)):
        """
        レザバーの状態更新
        変数名は2文字で構成される。
        1文字目は、rが参照クロック、uが入力、hがレザバー、yが出力を表す。
        2文字目は、xが[0,1]の連続値、sが{0,1}の2値、pが[-1,1]の連続値
        """
        self.NS = len(input)        # 外部時間ステップ数、データの長さに応じて設定
        self.NT = self.NS * self.NN # 内部時間ステップ数

        self.Up = input
        self.Dp = output

        if len(output) == 0: output = np.zeros((self.NS, self.Ny))

        hx = np.zeros(self.Nh)              # レザバー状態(x) [0,1]の連続値
        #hx = np.random.uniform(0,1,self.Nh)# レザバー状態(x) [0,1]の連続値　(ランダムに初期化)
        hs = np.zeros(self.Nh)              # レザバー状態(s) {0,1}の２値
        hc = np.zeros(self.Nh)              # 参照クロックに対する位相差を求めるためのカウンタ
        hp = np.zeros(self.Nh)              # [-1,1]の連続値
        ht = np.zeros(self.Nh)              # {-1,1}ラッチ動作
        #yp = np.zeros(self.Ny)
        #ys = np.zeros(self.Ny)
        #yx = np.zeros(self.Ny)
        if self.show: self.initialize_record()

        rs = 1
        count =0
        m = 0
        for n in tqdm(range(self.NT),disable=False):
            rs_prev = rs
            hs_prev = hs.copy()

            theta = np.mod(n/self.NN,1)     # 位相[0,1)
            rs = self.p2s(theta,0)          # 参照クロック
            us = self.p2s(theta,input[m])   # 入力信号を2値パルスにエンコード
            ds = self.p2s(theta,output[m])  # 
            #ys = self.p2s(theta,yp)

            sum = np.zeros(self.Nh)
            #sum += self.alpha_s*rs         # 参照クロックと同期させるための結合（ラッチ動作を用いない）
            sum += self.alpha_s*(hs-rs)*ht  # 参照クロックと同期させるための結合
            sum += self.Wi@(2*us-1)         # 入力
            sum += self.Wr@(2*hs-1)         # リカレント結合

            hsign = 1 - 2*hs                # xの変化方向（符号）
            hx = hx + hsign*(1.0+np.exp(hsign*sum/self.Temp))*self.dt # レザバー状態(x)の更新
            hs = np.heaviside(hx+hs-1,0)    # レザバー状態(s)の更新：(if x>1 then s=1, if x<0 then s=0)
            hx = np.clip(hx,0,1)            # レザバー状態(x)を[0,1]の範囲にクリップ
            hc[(hs_prev==1) & (hs==0)] = count # hsの立ち下がりでcountを保持
            #hc[(rs==1) and (hs==1)]+=1      # rsとhsのANDでカウント

            if (rs_prev==0 and rs==1) or (n == self.NT-1) :# 参照クロックの立ち上がり
                hp = 2*hc/self.NN-1         # デコード、カウンタの値を連続値に変換
                hc = np.zeros(self.Nh)      # カウンタをリセット
                ht = 2*hs-1                 # 参照クロック同期用ラッチ動作
                yp = self.Wo@hp             # 読み出し
                self.Hp[m]=hp
                self.Yp[m]=yp
                count = 0
                m += 1

            count += 1 # rsの立ち上がりからの経過をカウント

            if self.show: self.add_record(n,rs,hx,hs,us,ds)
        
        assert m == self.NS

        # オーバーフローを検出する。
        self.cnt_overflow=0
        for m in range(2,self.NS-1):
            tmp = np.sum( np.heaviside( np.fabs(self.Hp[m+1]-self.Hp[m]) - 0.6 ,0))
            self.cnt_overflow += tmp
        
        self.X = self.Hp # レザバー内部状態
        return self.Yp
    
    def initialize_record(self):
        self.Rs = np.zeros((self.NT, 1))
        self.Hx = np.zeros((self.NT, self.Nh))
        self.Hs = np.zeros((self.NT, self.Nh))
        self.Ys = np.zeros((self.NT, self.Ny))
        self.Us = np.zeros((self.NT, self.Nu))
        self.Ds = np.zeros((self.NT, self.Ny))
        self.Hp = np.zeros((self.NS, self.Nh))
        self.Yp = np.zeros((self.NS, self.Ny))
        
    def add_record(self,n,rs,hx,hs,us,ds):
        self.Rs[n]=rs
        self.Hx[n]=hx
        self.Hs[n]=hs
        self.Us[n]=us
        self.Ds[n]=ds

    def show_recode(self,):
        if not self.plot:
            return {},{},{},self.Hp[self.NS0:],self.Yp[self.NS0:],self.cnt_overflow
        else:
            return self.Us,self.Rs,self.Hx,self.Hp[self.NS0:],self.Yp[self.NS0:],self.cnt_overflow

    def plot(self,NS1=0,NS2=21):
        """
        NS1,NS2でプロット範囲の時刻を設定する。
        """
        NT1 = self.NN*NS1
        NT2 = self.NN*(NS2-1)

        itime = np.arange(self.NT)/self.NN  # internal time
        etime = np.arange(self.NS)          # external time

        Nr=5
        plt.figure(figsize=(16, 8))
        plt.subplot(Nr,1,1)
        plt.plot(etime[NS1:NS2],self.Up[NS1:NS2])
        plt.xlabel("time")
        plt.ylabel("Input (Up)")
        #plt.title("Input")

        plt.subplot(Nr,1,2)
        plt.plot(itime[NT1:NT2],self.Us[NT1:NT2])
        plt.plot(itime[NT1:NT2],self.Rs[NT1:NT2],"r:")
        plt.xlabel("time")
        plt.ylabel("Input (Up)")

        plt.subplot(Nr,1,3)
        plt.plot(itime[NT1:NT2],self.Hx[NT1:NT2])
        plt.plot(itime[NT1:NT2],self.Rs[NT1:NT2],"r:")
        plt.xlabel("time")
        plt.ylabel("Reservoir (Hx)")
        
        plt.subplot(Nr,1,4)
        plt.plot(etime[NS1:NS2],self.Hp[NS1:NS2])
        plt.xlabel("time")
        plt.ylabel("Reservoir (Hp)")

        plt.subplot(Nr,1,5)
        plt.plot(etime[NS1:NS2],self.Yp[NS1:NS2])
        plt.plot(etime[NS1:NS2],self.Dp[NS1:NS2])
        plt.xlabel("time")
        plt.ylabel("Output (Yp)")

        dir_name = "trashfigure"
        fig_name = "fig1"

        if self.c.savefig: plt.savefig("./{}/{}".format(dir_name,fig_name))
        if self.c.show: plt.show()