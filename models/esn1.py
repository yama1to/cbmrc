# Copyright (c) 2021-2022 Katori lab. All Rights Reserved
"""
基本的なESN (フィードバックなし)
"""
import numpy as np
from models.generate_matrix import *
c=None
def initialize():
    global Wr,Wb,Wi,Wo
    np.random.seed(int(c.seed))

    Wr = generate_random_matrix(c.Nx,c.Nx,c.alpha_r,c.beta_r,distribution="uniform",normalization="sr")
    Wi = generate_random_matrix(c.Nx,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    #Wb = generate_random_matrix(c.Nx,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wo = np.zeros(c.Nx * c.Ny).reshape((c.Ny, c.Nx))

def run(input,output=np.empty(0)):
    global X
    
    NS = len(input)
    X = np.zeros((NS, c.Nx))
    Y = np.zeros((NS, c.Ny))
    
    #x = np.random.uniform(-1, 1, Nx)/ 10**4
    x = np.zeros(c.Nx)
    for n in range(NS):
        u = input[n, :]
        x = np.tanh(Wi @ u + Wr @ x)
        y = Wo@x
        X[n,:] = x
        Y[n,:] = y
    
    return Y

def fit(input,output):
    assert len(input)==len(output)
    run(input)
    regression(X[c.NS0:],output[c.NS0:])
    
def regression(reservoir_state,output):
    global Wo
    M = reservoir_state
    G = output
    assert len(M)==len(G)

    ### Ridge regression
    if c.lambda0 == 0:
        Wo = np.dot(G.T,np.linalg.pinv(M).T)
    else:
        E = np.identity(c.Nx)
        TMP1 = np.linalg.inv(M.T@M + c.lambda0 * E)
        WoT = TMP1@M.T@G
        Wo =WoT.T
        
    assert Wo.shape == (c.Ny,c.Nx)

