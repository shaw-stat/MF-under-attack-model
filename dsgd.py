# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:50:10 2020

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
from functions import *

#%matplotlib inline

rho0 = 0.2
C = 10 # Number of factors
nbr_iter = 50# number of iterations
block_number = 4 # number of blocks to take from the matrix
sc= SparkContext.getOrCreate()
#mytest = np.loadtxt("D:\\新建文件夹2019\\SG_MCMC\\R3_test.txt", dtype=int)
mytest = np.loadtxt("D:\\新建文件夹2019\\SG_MCMC\\ua.test", dtype=int)
mytrain =np.loadtxt("D:\\新建文件夹2019\\SG_MCMC\\ua.base", dtype=int)
test_mean = mytest[:,2].mean()
train_mean = mytrain[:,2].mean()
mean_rate = (test_mean+train_mean)/2
# eta0 = 0.1
eta0=0.9
def SGD(R, Q, P, mask, Ni, Nj, blockRange):
    """
    This function is an implementation of the SGD algorithm described above.
    Input : R, Q, P, mask, Ni, Nj, blockRange
    Output : Q, P, n, blockRange
    """
    
    global rho0,eta0
    #eta = 0.01#first step size
    R_new = R.nonzero()
    n = R_new[0].size
    #eta=eta0
    rho=rho0
    
    for i in range(n):
        
        # if i% 10000 == 0:
        #     eta=eta0/(2**(i/10000))
        #     rho=rho0/(2**(i/10000))
        # if i<n:
        #     tau = i/n
        #     eta = eta0*(1-tau)+tau*0.01*eta0
        # else:
        #     eta = 0.01*eta0
        
        eta = eta0/(i+1)
        
        j = random.randint(0, n-1) # Pick randomly an element j
        row, col = R_new[0][j], R_new[1][j] # retrieve the row and column of the random j
        
        # take a small blocks from R, mask, Q and P
        Ri = R[row,col] 
        maski = mask[row,col]
        Qi = Q[row,:]
        Pi = P[:,col]
        
        # compute the gradient of Qi and Pi
        _, grad_Q = objective_Q(Pi, Qi, Ri, maski, rho)
        _, grad_P = objective_P(Pi, Qi, Ri, maski, rho)
        #eta = eta0 * (1 + i) ** (- 0.5)
        #eta=eta*0.96
        #eta=eta0
        
        # update the blocks of P and Q
        Q[row,:] = Qi - eta * grad_Q
        P[:,col] = Pi - eta * grad_P
        #print(np.linalg.norm(Q[row,:]))
        
    return (Q, P, n, blockRange)

def SGD2(R,mask, test,mask2):
    """
    This function is an implementation of the SGD algorithm described above.
    Input : R, Q, P, mask, Ni, Nj, blockRange
    Output : Q, P, n, blockRange
    """
    # Q = numpy.random.random_sample((R.shape[0], C))
    # P = numpy.random.random_sample((C, R.shape[1]))
    #Q =np.loadtxt('Q3_sgd_new3.csv',delimiter=',')
    #P =np.loadtxt('P3_sgd_new3.csv',delimiter=',')
    Q = np.ones([R.shape[0],C])*0.3
    P = np.ones([C,R.shape[1]])*0.3
    global eta0,rho0
    #eta = 0.01#first step size
    R_new = R.nonzero()
    n = R_new[0].size
    Rmse = []
    T=[]
    t0=time()
    eta=eta0
    rho=rho0
    
    for i in range(10000):
        if i<50000:
            tau = i/50000
            eta = eta0*(1-tau)+tau*0.01*eta0
        else:
            eta = 0.01*eta0
            
        
        #eta=eta0
        # if i% 20000 == 0:
        #     eta=eta0/(2**(i/20000))
        #     #rho=rho0/(2**(i/20000))
        #     #eta=eta0*(0.96**(i/10000))
        #     print("... iteration %s, eta %f,rho%f"%(i,eta,rho))
            
        
        j = random.randint(0, n-1) # Pick randomly an element j
        row, col = R_new[0][j], R_new[1][j] # retrieve the row and column of the random j
        
        # take a small blocks from R, mask, Q and P
        Ri = R[row,col]
        maski= mask[row,col]
        Qi = Q[row,:]
        Pi = P[:,col]
        
        # compute the gradient of Qi and Pi
        _, grad_Q = objective_Q(Pi, Qi, Ri, maski, rho)
        _, grad_P = objective_P(Pi, Qi, Ri, maski, rho)
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

def Parallelized_SGD(R, mask,test,mask2):
    """
    This function performs the Parallelized SGD algorithm
    Input : R, mask
    Output : Q, P
    """
    T=[]
    t0=time()
    
    global nbr_iter, block_number, C,eta0,rho0
    
    # Q = np.ones([R.shape[0],C])*0.5
    # P = np.ones([C,R.shape[1]])*0.5
    Q = numpy.random.random_sample((R.shape[0], C))
    P = numpy.random.random_sample((C, R.shape[1]))
    #Q =np.loadtxt('Q3_sgd5.csv',delimiter=',')
    #P =np.loadtxt('P3_sgd5.csv',delimiter=',')
    block_i = (int(R.shape[0]/block_number), int(R.shape[1]/block_number))
    
    
    rowRangeList = [[k*block_i[0],(k+1)*block_i[0]] for k in range(block_number)]
    colRangeList = [[k*block_i[1],(k+1)*block_i[1]] for k in range(block_number)]

    rowRangeList[-1][1] += R.shape[0]%block_number
    colRangeList[-1][1] += R.shape[1]%block_number

    Rmse = []

    for iter_ in range(nbr_iter):
        
        if iter_ % 10 == 0:
            print("... iteration %s"%(iter_))
        
        for epoch in range(block_number):
            grid = []
            
            
            for block in range(block_number):
                rowRange = [int(rowRangeList[block][0]), int(rowRangeList[block][1])]
                colRange = [int(colRangeList[block][0]), int(colRangeList[block][1])]
                
                
                # The subsamples in each matrix and vector
                Rn = R[rowRange[0]:rowRange[1], colRange[0]:colRange[1]]
                maskn = mask[rowRange[0]:rowRange[1], colRange[0]:colRange[1]]
                Qn = Q[rowRange[0]:rowRange[1],:]
                Pn = P[:,colRange[0]:colRange[1]]
                
                Ni = {}
                for i in range(rowRange[0],rowRange[1]):
                    Ni[int(i-int(rowRange[0]))] = R[i,:].nonzero()[0].size
                    
                Nj = {}
                for i in range(colRange[0],colRange[1]):
                    Nj[i-colRange[0]] = R[:,i].nonzero()[0].size 
                    
                if (Rn.nonzero()[0].size != 0):
                    grid.append([Rn, Qn, Pn, maskn, Ni, Nj, (rowRange, colRange)])
                    
                    
                    
            rdd = sc.parallelize(grid, block_number).\
                        map(lambda x: SGD(x[0],x[1],x[2],x[3],x[4],x[5],x[6])).collect()
                
                
            for elem in rdd:
                rowRange,colRange = elem[3]
                Q[rowRange[0]:rowRange[1],:] = elem[0]
                P[:,colRange[0]:colRange[1]] = elem[1]

            colRangeList.insert(0,colRangeList.pop())
            
        nuser = test.shape[0]
        nitem = test.shape[1]
        
        pre = np.dot(Q[:nuser,:], P[:,:nitem])
        #pre[np.where((pre>0)&(pre<1))] = 1
        #pre[np.where(pre>5)] = 5
        
        temp = mask2*(test-pre)
        rows, cols = np.nonzero(temp)
        Rmse.append(np.sqrt(np.power(temp[rows,cols],2).mean()))     
        T.append(time()-t0)    
            
            
    return Q,P,Rmse,T

def outputMatrix(A, path):
    """
    This function outputs a matrix to a csv file
    """
    f = open(path, 'w', 100)
    rows= A.shape[0]
    cols = A.shape[1]
    for row in range(rows):
        for col in range(cols):  
            if col == cols-1:
                f.write(str(A[row,col])) 
            else:
                f.write(str(A[row,col]) + ",")
        f.write("\n")

    f.flush()
    f.close()
    

def load_data(filename="u.data",scale = True,small_data=False):
    """
    This function returns :
        R : the matrix user-item containing the ratings
        mask : matrix is equal to 1 if a score existes and 0 otherwise
        
    """
    global mean_rate
    
    data = np.loadtxt(filename, dtype=int)[:,:3]
    #data = data_norm(data0)
    
    
    if filename=="D:\\新建文件夹2019\\SG_MCMC\\ua.base":
        R = sparse.csr_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)),dtype=float)
        
    else:
        R = sparse.csr_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)),dtype=float)
    mask = sparse.csr_matrix((np.ones(data[:, 2].shape),(data[:, 0]-1, data[:, 1]-1)), dtype=bool )
    
    # #normalization
    # R= (R - np.mean(R, axis=0)) 
    # R= (R - np.mean(R, axis=1)) / np.std(R, axis=1)
     
    # take a small part of the whole data for testing 
    
    
     
    
    
    
    if scale==True:
        if filename=="D:\\新建文件夹2019\\SG_MCMC\\ua.base":
            R = np.loadtxt('R_a_base_scale.txt',delimiter=',')
            mask = sparse.csr_matrix((np.ones(R.nonzero()[0].shape[0]),(R.nonzero()[0], R.nonzero()[1])), dtype=bool )
        elif filename=="D:\\新建文件夹2019\\SG_MCMC\\ua.test":
            R = np.loadtxt('R_a_test_scale.txt',delimiter=',')
            mask = sparse.csr_matrix((np.ones(R.nonzero()[0].shape[0]),(R.nonzero()[0], R.nonzero()[1])), dtype=bool )
        else:
            print('not scaling')
            
    if small_data == True:
        R = (R[0:100, 0:100].copy())
        mask = (mask[0:100, 0:100].copy())
    
    
    # R = R.toarray()
    # mask = mask.toarray()
    
    
    return R, mask


def data_norm(data,mode):
    f_data = pd.DataFrame(data)
    if mode==1:
        data[:,2] = data[:,2]-np.mean(data[:2])
        
    return data

def scale_sparse_vector(x):
    if x[x!=0].shape[0]>0:
        x[x!=0]=scale(x[x!=0],with_mean=True,with_std=True)
    
def scale_matrix(R):
    RR = R.copy()
    d_R = pd.DataFrame(RR)
    d_R.apply(scale_sparse_vector,axis=1)
    return np.array(d_R)
    