
import numpy as np
import matplotlib.pyplot as plt
from .generate_data_sequence_ou import OUdataset

def generate_narma_ou(N,delay,seed=0):
    N = N + 200
    _, u = OUdataset(N)
    u = u/np.max(u)/2
    
    # Generate NARMA sequence
    d = np.zeros((N))
    for i in range(N-1):
        d[i+1] = 0.3*d[i] + 0.05*d[i] * \
            np.sum(d[i-9:i+1]) + 1.5*u[i-9]*u[i] + 0.1
    d = d[200:]
    u = u[200:]
    N = N-200
    if np.isfinite(d).all():
        u = u.reshape((N,1))
        d = d.reshape((N,1))
        return u,d
    else:
        print("again")
        return generate_narma_ou(N,delay,seed=seed+1)


if __name__ == '__main__':
    u,d = generate_narma_ou(1000)
    plt.plot(u)
    plt.plot(d)
    plt.show()
    print(u.shape,d.shape)