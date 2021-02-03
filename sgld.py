# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:55:56 2020

@author: shaw
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from savefile import *
from rmse import *
from time import time

at_size = 0.3
fil_size = 0.5
it = 1


#Initialize the parameters
def init_para(alpha,beta,D,L,M):
    #LU=np.diag(np.random.gamma(alpha,beta,D))
    #LV=np.diag(np.random.gamma(alpha,beta,D))
    LU=np.diag(2*np.ones(D))
    LV=np.diag(2*np.ones(D))
    #U=np.random.multivariate_normal(mean=np.zeros(D), cov=np.linalg.pinv(LU), size=L)
    #V=np.random.multivariate_normal(mean=np.zeros(D), cov=np.linalg.pinv(LV), size=M)
    U = np.ones([L,D])*0.3
    V = np.ones([M,D])*0.3
    # U = np.random.random_sample((L,D))
    # V = np.random.random_sample((M,D))
    paras={'U':U,'V':V,'LambdaU':LU,'LambdaV':LV}
    return paras

# Calculate the stochastic gradient
def gun(x,U,V):
    gu = np.zeros(U.shape[1]+1)
    gu[0]=int(x[0])
    gu[1:,]=(x[2]-U[int(x[0])-1].dot(V[int(x[1])-1]))*V[int(x[1])-1]
    return gu

def gvn(x,U,V):
    gv = np.zeros(U.shape[1]+1)
    gv[0]=int(x[1])
    gv[1:,]=(x[2]-U[int(x[0])-1].dot(V[int(x[1])-1]))*U[int(x[0])-1]
    return gv
    

def generate_gradient(U,V,data):
    #create dataframe to apply function
    df = pd.DataFrame(data)
    mini = data.shape[0]
    
    gu_temp = df.apply(gun,axis=1,args=(U,V))
    gu_temp = pd.DataFrame(np.array(list(gu_temp)))
    gv_temp = df.apply(gvn,axis=1,args=(U,V))
    gv_temp = pd.DataFrame(np.array(list(gv_temp)))
    
    GU = np.array(gu_temp.groupby(0).sum()/mini)
    GV = np.array(gv_temp.groupby(0).sum()/mini)
    
    return GU,GV

# Generate the data
def generate_data(mini,myscale):
    
    if (myscale == True):
        #R_train = np.loadtxt('R_a_base_scale.txt',delimiter=',')
        R_train = np.loadtxt('R_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
        R_test = np.loadtxt('R_a_test_scale.txt',delimiter=',')
        data= matrix_list(R_train)
        test= matrix_list(R_test)
        print('the data has been scaled')
    elif (myscale ==False):
        test = np.loadtxt('D:\\新建文件夹2019\\SG_MCMC\\ml-100k\\ml-100k\\ua.test').astype(int)
        #test = np.loadtxt('D:\\新建文件夹2019\\SG_MCMC\\R3_test.txt').astype(int)
        test = test[:,:3]
        data = np.loadtxt('D:\\新建文件夹2019\\SG_MCMC\\ml-100k\\ml-100k\\ua.base').astype(int)
        #data = np.loadtxt('D:\\新建文件夹2019\\SG_MCMC\\R3_train.txt').astype(int)
        data = data[:,:3]
        print('the data has not been scaled')
    
    N = data.shape[0]
    mini = int(N*mini)
    L=int(data[:,0].max())
    M=int(data[:,1].max()) 
    
    # generate hi
    hi = np.zeros(L)
    hj = np.zeros(M)
    for i in range(L):
        hi[i] = 1-(1-data[:,0].tolist().count(i+1)/N)**mini
    
    for j in range(M):
        hj[j] = 1-(1-data[:,1].tolist().count(j+1)/N)**mini

    return data,test,hi,hj,L,M

def matrix_list(R):
    R_n0 = R.nonzero()
    N = R_n0[0].size
    data = np.zeros([N,3])
    for i in range(N):
        xi = R_n0[0][i]
        yi = R_n0[1][i]
        data[i,0]=xi+1
        data[i,1]=yi+1
        data[i,2]=R[xi,yi]
    
    return data
    

def sgld_worker(params,data,hi,hj,lround,minibatchsize,step0):
    U1 = params['U']
    V1 = params['V']
    LU = params['LambdaU']
    LV = params['LambdaV']
    N=data.shape[0]
    D=U1.shape[1]
    T=[]
    t0 = time()
    stept = step0
    
    
    
    
    #'update parameters'
    for t in range(lround):
        # if ((t+1)%50==0):
        #     stept = step0*(1+t/500)**(-0.51)
        stept = step0*(1+t/500)**(-0.51)
        #stept=step0
        #'generate the minibatch in dataset of each worker'
        batch=data[np.random.choice(np.arange(N),int(minibatchsize*N),replace=False),:]
        #'generate gradient of log likelihood'
        uindex=list(np.sort((np.unique(batch[:,0])-1)).astype(int))
        vindex=list(np.sort((np.unique(batch[:,1])-1)).astype(int))
        
        HI=np.diag(1/hi[uindex])
        HJ=np.diag(1/hj[vindex])
        
        GU,GV=generate_gradient(U1,V1,batch)            #function1
        
        # vtu=np.random.multivariate_normal(mean=np.zeros(D), cov=step*np.identity(D), size=len(uindex))
        # vtv=np.random.multivariate_normal(mean=np.zeros(D), cov=step*np.identity(D), size=len(vindex))
        random = np.random.multivariate_normal(mean=np.zeros(D), cov=stept*np.identity(D))
        vtu = np.tile(random,(len(uindex),1))
        vtv = np.tile(random,(len(vindex),1))
        #'update feature vector'
        U1[uindex,:]=U1[uindex,:]+0.5*stept*(N*GU-np.dot(np.dot(HI,U1[uindex,:]),LU))+vtu
        V1[vindex,:]=V1[vindex,:]+0.5*stept*(N*GV-np.dot(np.dot(HJ,V1[vindex,:]),LV))+vtv
        
        # if (t>=149)&((t+1)%50==0):
        # #if ((t+1)%10==0):
        #     print ('update hyperparameter')
        #     for d in range (D):
        #             normU=np.linalg.norm(U1[:,d],ord=2)**2
        #             normV=np.linalg.norm(V1[:,d],ord=2)**2
        #             params['LambdaU'][d,d]=np.random.gamma(alpha+0.5*L,beta+0.5*normU,1)
        #             params['LambdaV'][d,d]=np.random.gamma(alpha+0.5*M,beta+0.5*normV,1) 
                    # params['LambdaU'][d,d]=np.random.gamma(beta+0.5*normU,alpha+0.5*L,1)
                    # params['LambdaV'][d,d]=np.random.gamma(beta+0.5*normV,alpha+0.5*M,1)
                    
        #filename= 'sgld_d10_ua_0.51_time%d' %(t)
        filename = 'sgld_atk_avg_%f_%f_%d_time%d'%(at_size,fil_size,it,t)
        saveres('.\dat2', filename, params, ext = 'npy', verbose = True)
        T.append(time()-t0)
        
    return U1,V1,T

def rmse(x,params):
    pre = params['U'][int(x[0])-1].dot(params['V'][int(x[1])-1])
    # if pre>5:
    #     pre=5
    # if (pre>0)&(pre<1):
    #     pre=1
    
    return (x[2]-pre)**2

def predic_err(test,T,f):
    test = pd.DataFrame(test)
    Rmse = np.zeros(T)
    for t in range(T):
        filename = f+'time%d.npy' %(t)
        para = np.load(filename).tolist()
        R = test.apply(rmse,axis=1,args=(para,))
        Rmse[t]=np.sqrt(R.mean())
        
    
    return Rmse


if __name__ == "__main__":
    alpha=1
    beta=300
    D=10
    # L=943
    # M=1682                                                                                                                                                                  
    lround=400                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    mini=0.05
    step0=1e-4
    data,test,hi,hj,L,M = generate_data(mini,myscale = True)
    paras = init_para(alpha,beta,D,L,M)
    time0 = time()
    U,V,T = sgld_worker(paras,data,hi,hj,lround,mini,step0)
    print ('Process success in %d seconds'%(time()-time0))
    #Rmse = predic_err(test,lround,'./dat2/sgld_d10_ua_0.51_')
    Rmse = predic_err(test,lround,'./dat2/sgld_atk_avg_%f_%f_%d_'%(at_size,fil_size,it))
    #np.savetxt('./dat2/Rmse_sgld_2.npy',Rmse)
    #np.savetxt('./dat2/T_sgld_2.npy',T)
    np.savetxt('Rmse_sgld_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),Rmse)
    np.savetxt('T_sgld_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),T)
    
    
    
    
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('Rmse')
    plt.plot(T,Rmse)
    #plt.savefig('Rmse_T_sgld.png')
    #print('the time is %f seconds and the minimum of Rmse is %f，the step size is %f'%(np.array(T).max(),np.array(Rmse).min(),step0))