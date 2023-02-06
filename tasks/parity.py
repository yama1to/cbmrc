from generate_datasets.generate_data_sequence import generate_parity
import numpy as np
import matplotlib.pyplot as plt

def dataset(MM1,MM2,tau=4,k=3):
    T = MM1 +(tau+k-1)#+MM2 
    U1,D1 = generate_parity(T,tau,k)
    T = MM2 +(tau+k-1)#+MM2 
    U2,D2 = generate_parity(T,tau,k)
    # plt.plot(D,marker="o")
    # plt.tight_layout()
    # plt.savefig("parity-d.eps")
    U1 = np.tanh(U1)
    U2 = np.tanh(U2)

    return U1,D1,U2,D2



def evaluate(Yp,Dp,MM,tau=4,k=3):
    # 評価（ビット誤り率, BER）
    train_Y_binary = np.zeros(MM-tau-k+1)
    Yp = Yp[tau+k-1:]
    Dp = Dp[tau+k-1:]
    #閾値を0.5としてバイナリ変換する
    for n in range(MM-tau-k+1):
        train_Y_binary[n] = np.heaviside(Yp[n]-np.tanh(0.5),0)
    
    BER = np.linalg.norm(train_Y_binary-Dp[:,0], 1)/(MM-tau-k+1)
    print('BER ={:.3g}'.format(BER))
    return BER


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
