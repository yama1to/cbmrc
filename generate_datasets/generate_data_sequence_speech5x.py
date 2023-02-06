# Copyright (c) 2021-2022 Katori lab. All Rights Reserved
"""
TI46音声データからcochleagramの時系列とクラス分類のターゲット(one-hot-vector)時系列を生成する。
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import wave 
import random #リストのshuffleで使用する
from lyon.calc import LyonCalc

path_wav = "generate_datasets/ti46wave/" # TI46の音声データを配置するディレクトリ
save_images = False

def save_wave(wave,filename):
    p = path + "images/"
    if not os.path.isdir(p):os.makedirs(p)
    file = p + "wave_"+filename+".eps"
    plt.plot(wave)
    plt.savefig(file)
    plt.clf()
    plt.close()

def save_coch(c,filename):
    p = path + "images/"
    if not os.path.isdir(p):os.makedirs(p)
    file = p + "coch_"+filename+".eps"
    plt.imshow(c.T)
    plt.savefig(file)
    plt.clf()
    plt.close()

def cut(wave):
    len_output=10000 # 出力波形の長さ
    count = 0
    start = 0
    for i in range(len(wave)):
        if np.fabs(wave[i]) > 200:
            count += 1
            if count==10:
                start = i
                break 

    start = i-2500
    start = max(start,0)
    end = min(len_output+start,wave.shape[0])
    output = np.zeros(len_output)
    output[:end-start] = wave[start:end]
    return output

def generate_target(dataset,Nclass,Nstep):
    # 分類クラスのone-hot-vector時系列を生成する。
    # Nclass：時系列の次元、Nstep:1データあたりの時系列の長さ
    Ndata = len(dataset)
    target = np.zeros((Ndata*Nstep,Nclass))
    start = 0
    for d in dataset:
        target[start:start + Nstep,d[0]] = 1
        start += Nstep            
    return target

def generate_cochleagram(dataset,cl):
    calc = LyonCalc()
    coch = np.empty((0,0))
    for d in dataset:
        filename = d[4]
        wave = load_wave(filename)
        c = calc.lyon_passive_ear(wave, sample_rate=cl["sample_rate"], decimation_factor=cl["decimation_factor"], ear_q=cl["ear_q"], step_factor=cl["step_factor"], tau_factor=cl["tau_factor"])      
        coch = np.append(coch,c,axis=0) if not coch.shape == (0,0) else c
        #print(coch.shape)
        if save_images:
            save_coch(c,filename)
    return coch,c.shape # 連結したcochleagramと、データ１つ分のcochleagramのshapeを返す

def load_wave(filename):
    file = path_wav + filename + ".wav"
    with wave.open(file,mode='r') as W:
        W.rewind()
        buf = W.readframes(-1)  # read allA
        #16bitごとに10進数
        wa = np.frombuffer(buf,dtype='int16').astype(np.float64)
    wa = cut(wa)
    return wa

def load_waves(dataset):
    waves = np.empty((0,0))
    for d in dataset:
        filename = d[4]
        w = load_wave(filename)
        waves = np.append(waves,w,axis=0) if not waves.shape == (0,0) else w
        if save_images:
            save_wave(w,filename)
    return waves

def save_config(dataset_train,dataset_test,shape,config_lyon):
    # データセットと設定パラメータをcsvに保存する
    if not os.path.isdir(path):os.makedirs(path)
    file = path + "data.csv"
    f = open(file,"w")
    #(len(dataset_train),len(dataset_test),c_shape[0],c_shape[1],Nclass)
    ### 訓練データの数、テストデータの数、コクリアグラムの時間ステップ数、周波数成分の数、ターゲットの次元
    f.write("len(dataset_train),{},\n".format(shape[0]))
    f.write("len(dataset_test),{},\n".format(shape[1]))
    f.write("Nstep,{},\n".format(shape[2]))
    f.write("frequency components,{},\n".format(shape[3]))
    f.write("Nclass,{},\n".format(shape[4]))
    ### lyon filter
    f.write("sample_rate,{},\n".format(config_lyon["sample_rate"]))
    f.write("decimation_factor,{},\n".format(config_lyon["decimation_factor"]))
    f.write("ear_q,{},\n".format(config_lyon["ear_q"]))
    f.write("step_factor,{},\n".format(config_lyon["step_factor"]))
    f.write("tau_factor,{},\n".format(config_lyon["tau_factor"]))

    for d in dataset_train:
        line = "train,{},{},{},{},{},\n".format(d[0],d[1],d[2],d[3],d[4])
        f.write(line)

    for d in dataset_test:
        line = "test,{},{},{},{},{},\n".format(d[0],d[1],d[2],d[3],d[4])
        f.write(line)
    
    f.close()

def config_dataset_speech5a():
    ### 基本的な数字の識別（崎野の設定、数字でソートしている）
    path = "generate_datasets/speech5a/"
    prompt = ["00","01","02","03","04","05","06","07","08","09"]
    speaker = ["f1","f2","m2","m3","m5"]
    sets = ["set0","set1","set2","set3","set4","set5","set6","set7","set8","set9"]
    # sets は sessionとtokenに対応する。要確認
    Nclass = 10
    dataset_train=[]
    dataset_test=[]
    for p in prompt:
        dataset=[]
        for s in speaker:
            for t in sets:
                c = prompt.index(p) #数字の区別をクラス分類のターゲットに設定
                filename="{}{}{}".format(p,s,t)
                dataset.append((c,p,s,t,filename))
        random.shuffle(dataset)
        dataset_train.extend(dataset[:25])
        dataset_test.extend(dataset[25:])
    #print(dataset_test)

    num_train = np.zeros((10))
    num_test = np.zeros((10))
    for i in range(250):
        num_train[dataset_train[i][0]] += 1
        num_test[dataset_test[i][0]] += 1
    print('train:',num_train)
    print('test:',num_test)


    config_lyon={"sample_rate":12500, "decimation_factor":200, "ear_q":3, "step_factor":0.091, "tau_factor":3}
    return path,dataset_train,dataset_test,Nclass,config_lyon

def config_dataset_speech5b():
    ### 話者の識別
    path = "generate_datasets/speech5b/"
    prompt = ["00","01","02","03","04","05","06","07","08","09"]
    speaker = ["f1","f2","m2","m3","m5"]
    sets = ["set0","set1","set2","set3","set4","set5","set6","set7","set8","set9"]
    Nclass = 5
    dataset=[]
    for p in prompt:
        for s in speaker:
            for t in sets:
                c = speaker.index(s) #話者の区別をクラス分類のターゲットに設定
                filename="{}{}{}".format(p,s,t)
                dataset.append((c,p,s,t,filename))
    random.shuffle(dataset)
    dataset_train = dataset[:250]
    dataset_test = dataset[250:]
    
    config_lyon={"sample_rate":12500, "decimation_factor":200, "ear_q":3, "step_factor":0.091, "tau_factor":3}
    return path,dataset_train,dataset_test,Nclass,config_lyon

def config_dataset_speech5c():
    ### 数字の識別、500個のデータをシャッフルして半分に分ける
    path = "generate_datasets/speech5c/"
    prompt = ["00","01","02","03","04","05","06","07","08","09"]
    speaker = ["f1","f2","m2","m3","m5"]
    sets = ["set0","set1","set2","set3","set4","set5","set6","set7","set8","set9"]
    Nclass = 10
    dataset=[]
    for p in prompt:
        for s in speaker:
            for t in sets:
                c = prompt.index(p) #数字の区別をクラス分類のターゲットに設定
                filename="{}{}{}".format(p,s,t)
                dataset.append((c,p,s,t,filename))
    random.shuffle(dataset)
    dataset_train = dataset[:250]
    dataset_test = dataset[250:]

    num_train = np.zeros((10))
    num_test = np.zeros((10))
    for i in range(250):
        num_train[dataset_train[i][0]] +=1
        num_test[dataset_test[i][0]] +=1
    print('train:',num_train)
    print('test:',num_test)

    config_lyon={"sample_rate":12500, "decimation_factor":200, "ear_q":3, "step_factor":0.091, "tau_factor":3}
    return path,dataset_train,dataset_test,Nclass,config_lyon

def generate_dataset(data):
    global path

    ### datasetの設定
    if data==1:
        path,dataset_train,dataset_test,Nclass,config_lyon = config_dataset_speech5a()
    elif data==2:
        path,dataset_train,dataset_test,Nclass,config_lyon = config_dataset_speech5b()
    else:
        path,dataset_train,dataset_test,Nclass,config_lyon = config_dataset_speech5c()

    ### waveデータの読み込み
    waves_train = load_waves(dataset_train)
    waves_test = load_waves(dataset_test)
    
    ### コクリアグラムの生成
    c_shape=(0,0)
    coch_train,c_shape = generate_cochleagram(dataset_train,config_lyon)
    coch_test,c_shape = generate_cochleagram(dataset_test,config_lyon)
    
    ### クラス分類のターゲット(one-hot-vector)時系列を生成
    target_train = generate_target(dataset_train,Nclass,c_shape[0])
    target_test = generate_target(dataset_test,Nclass,c_shape[0])

    ### データの出力・保存
    shape = (len(dataset_train),len(dataset_test),c_shape[0],c_shape[1],Nclass)    
    print(shape)#訓練データの数、テストデータの数、コクリアグラムの時間ステップ数、周波数成分の数、ターゲットの次元
    save_config(dataset_train,dataset_test,shape,config_lyon)
    np.savez(path+"data",c_train = coch_train,c_test = coch_test,t_train = target_train,t_test = target_test,shape=shape)
    load_dataset(data)

def load_dataset(data):
    ### データ読み込みのテスト
    if data==1:
        name = 'speech5a'
    elif data==2:
        name = 'speech5b'
    else:
        name = 'speech5c'
        
    f = np.load("generate_datasets/"+name+"/data.npz")
    U1 = np.array(f['c_train'])
    U2 = np.array(f['c_test'])
    D1 = np.array(f['t_train'])
    D2 = np.array(f['t_test'])
    e = np.array(f['shape'])

    return U1,U2,D1,D2,e
    # print(a,a.shape,e)
    # plt.plot(c)
    # plt.show()

if __name__ == "__main__":
    generate_dataset()
    #load_dataset()