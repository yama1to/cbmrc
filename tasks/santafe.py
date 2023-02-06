from generate_datasets.generate_data_sequence_santafe import generate_santafe
import numpy as np
import matplotlib.pyplot as plt

def dataset(MM1,MM2,future = [1,2,3,4,5]):
    U1,D1,U2,D2,normalize = generate_santafe(future = future,train_num = MM1,test_num =MM2,)
    U1 /= normalize
    U2 /= normalize
    return U1,D1,U2,D2,normalize



def evaluate(Yp,Dp,normalize,MM,):
    sum=0
    Yp = Yp*normalize
    Dp = Dp*normalize
 
    for j in range(MM):
        sum += (Yp[j] - Dp[j])**2
    MSE = sum/MM
    RMSE = np.sqrt(MSE)
    
    NRMSE = RMSE/np.var(Dp)#np.std(Dp)#np.var(Dp)

    NMSE =(MSE/np.var(Dp))
    
    NMSE = np.sum(NMSE)/NMSE.size
    NRMSE = np.sum(NRMSE)/NRMSE.size

    print('NMSE ={:.3g}'.format(NMSE))
    print('NRMSE ={:.3g}'.format(NRMSE))
    return NMSE,NRMSE


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
