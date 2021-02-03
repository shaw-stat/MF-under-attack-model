# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 21:47:49 2021

@author: shaw
"""


import sys
import math
from time import time
import random
import csv
import numpy
from pyspark import SparkContext
from scipy import sparse
from sklearn.preprocessing import normalize,scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dsgd import load_data,outputMatrix

import numpy as np

rho0 = 0.5
C = 10 # Number of factors
eta0=0.001

def objective(P, Q, R, mask, rho, Ni, Nj):
    
    r = (R - Q.dot(P)) * mask

    val = np.sum(r ** 2)/2. + rho /2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_Q = -Ni*np.dot(Q.T,r)/(np.sqrt((Nj*r)**2)) + rho * Q

    grad_P = -Nj*np.dot(r, P.T)/(np.sqrt((Nj*r)**2)) + rho * P

    return val, grad_P, grad_Q

    
def objective_Q(P0, Q, R, mask, rho,Ni,Nj):
    """
    This function returns two values : 
        -The value of the objective function at P0 fixed 
        -The value of the gradient of Q 
    """
    val,_ ,grad_Q = objective(P0, Q, R, mask, rho,Ni, Nj)

    return (val, grad_Q)


def objective_P(P, Q0, R, mask, rho, Ni, Nj):
    """
    This function returns two values : 
        -The value of the objective function at Q0 fixed 
        -The value of the gradient of P
    """
    val, grad_P,_ = objective(P, Q0, R, mask, rho,Ni, Nj)

    return (val, grad_P)

def SGD_R1(R,mask, test,mask2):
    Q = np.ones([R.shape[0],C])*0.5
    P = np.ones([C,R.shape[1]])*0.5
    global eta0,rho0
    #eta = 0.01#first step size
    R_new = R.nonzero()
    n = R_new[0].size
    Rmse = []
    T=[]
    t0=time()
    eta=eta0
    rho=rho0
    
    for i in range(5000):
        if i<50000:
            tau = i/50000
            eta = eta0*(1-tau)+tau*0.01*eta0
        else:
            eta = 0.01*eta0
        
        j = random.randint(0, n-1) # Pick randomly an element j
        row, col = R_new[0][j], R_new[1][j] # retrieve the row and column of the random j
        
        # take a small blocks from R, mask, Q and P
        Ri = R[row,col]
        maski= mask[row,col]
        Qi = Q[row,:]
        Pi = P[:,col]
        
        # compute the gradient of Qi and Pi
        _, grad_Q = objective_Q(Pi, Qi, Ri, maski, rho, len(R[row,:].nonzero()[0]),len(R[:,col].nonzero()[0]))
        _, grad_P = objective_P(Pi, Qi, Ri, maski, rho,len(R[row,:].nonzero()[0]),len(R[:,col].nonzero()[0]))
        #eta = eta0 * (1 + i) ** (- 0.5)
        #eta=eta*0.96
        #eta=eta0
        #if ((t>0)and(Rmse<))
        
        # update the blocks of P and Q
        Q[row,:] = Qi - eta * grad_Q
        P[:,col] = Pi - eta * grad_P
        #print(np.linalg.norm(Q[row,:]))
        
        nuser = test.shape[0]
        nitem = test.shape[1]
        
        pre = np.dot(Q[:nuser,:], P[:,:nitem])
        #pre[np.where((pre>0)&(pre<1))] = 1
        #pre[np.where(pre>5)] = 5
        
        temp = mask2*(test-pre)
        rows, cols = np.nonzero(temp)
        Rmse.append(np.sqrt(np.power(temp[rows,cols],2).mean()))
        T.append(time()-t0)
        
    return (Q, P, Rmse,T)



if __name__=="__main__":
    at_size = 0
    fil_size = 0
    it = 1

global R, P, Q
output_Q = "Q_sgdr1_atk_avg_%f_%f_%d.csv"%(at_size,fil_size,it)
output_P = "P_sgdr1_atk_avg_%f_%f_%d.csv"%(at_size,fil_size,it)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

#load data
R, mask = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.base" , scale = True,small_data=False)
# R = np.loadtxt('R_atk_rand_%f_%f_%d.npy'%(at_size,fil_size,it))
# mask = np.loadtxt('mask_atk_rand_%f_%f_%d.npy'%(at_size,fil_size,it))
test, mask2 = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.test" ,scale = True ,small_data=False)
mask2 = mask2.toarray()
t = time()
print("Start Process ....")
Q_sgd,P_sgd,Rmse_sgd,T_sgd=SGD_R1(R,mask,test,mask2)
print("Process finished in %s s"%(time()-t))
# Wrtie the obtained Matrices to csv file
outputMatrix(P_sgd,output_P)
outputMatrix(Q_sgd,output_Q)
print(time()-t)
#np.savetxt('Rmse_dsgd_normal_2.npy',Rmse_dsgd)
#np.savetxt('T_dsgd_normal_2.npy',T_dsgd)
np.savetxt('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),Rmse_sgd)
np.savetxt('T_sgd__atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),T_sgd)

plt.figure()
plt.xlabel('time')
plt.ylabel('Rmse')
plt.plot(T_sgd,Rmse_sgd,'g')
plt.title('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
print('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))

