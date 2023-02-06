from generate_datasets.generate_data_sequence_speech5x import generate_dataset, load_dataset
import numpy as np
import matplotlib.pyplot as plt

def dataset(data,load):
    #train_coch,valid_coch ,train_target, valid_target, SHAPE
    if load:
        train_coch,valid_coch ,train_target, valid_target, shape = load_dataset(data)
    else:
        train_coch,valid_coch ,train_target, valid_target, shape = generate_dataset(data)
    
    (num_train,num_test,num_steps_cochlear,num_freq,Nclass) = shape
    print(shape)#訓練データの数、テストデータの数、コクリアグラムの時間ステップ数、周波数成分の数、ターゲットの次元
    train_coch,valid_coch ,train_target, valid_target\
        = train_coch[:num_train*num_steps_cochlear],valid_coch[:num_test*num_steps_cochlear] ,\
            train_target[:num_train*num_steps_cochlear], valid_target[:num_test*num_steps_cochlear]

    normalize = max(np.max(train_coch),np.max(valid_coch))
    train_coch,valid_coch = train_coch/normalize ,valid_coch/normalize

    return train_coch,train_target, valid_coch ,valid_target, shape



def evaluate(Y,D,dataset_num,mode='train'):
    WSR = 1 - np.sum(abs(Y-D)/2)/dataset_num
    if mode=='train':
        print("train:",WSR)
    else:
        print("test:",WSR)

    return WSR



def run_model(c,model,UP,DP,Ndata,Nstep,mode="test"):
    from tqdm import tqdm
    """
    UP: 入力信号の時系列
    DP: ターゲットの時系列
    Ndata:データ（発話）の数
    Nstep:１つのデータ（発話）あたりの時間ステップ数
    mode:{test,train}
    """
    collect_state_matrix = np.empty((UP.shape[0],c.Nx))
    start = 0
    for _ in tqdm(range(Ndata),disable=c.do_not_use_tqdm):
        Dp = DP[start:start + Nstep]
        Up = UP[start:start + Nstep]
        model.run(input=Up,output=Dp)
        collect_state_matrix[start:start + Nstep,:] = model.X
        start += Nstep


    if mode == "train":
        M = collect_state_matrix
        G = DP.copy()
        model.regression(M,G)

    Y_pred = collect_state_matrix @ model.Wo.T 

    prediction = np.zeros((Ndata,c.Ny))
    start = 0
    for i in range(Ndata):
        tmp = Y_pred[start:start+Nstep,:]  # 1つのデータに対する出力
        max_index = np.argmax(tmp, axis=1) # 最大出力を与える出力ノード番号
        histogram = np.bincount(max_index) # 出力ノード番号のヒストグラム
        idx = np.argmax(histogram)
        prediction[i][idx] = 1             # 最頻値
        start = start + Nstep

    dp = [DP[i] for i in range(0,UP.shape[0],Nstep)]
    return prediction,dp




def plot(c,Up,Hp,Yp,Dp,dir_name = "trashfigure",fig_name="mc1"):
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
    
    if c.savefig:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if c.show :plt.show()
