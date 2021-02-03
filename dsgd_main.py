# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:45:49 2020

@author: shaw
"""


from dsgd import *
import matplotlib.pyplot as plt

at_size = 0.3
fil_size = 0.5
it = 1

global R, P, Q
sc= SparkContext.getOrCreate()
output_Q = "Q_sgd_atk_avg_%f_%f_%d.csv"%(at_size,fil_size,it)
output_P = "P_sgd_atk_avg_%f_%f_%d.csv"%(at_size,fil_size,it)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

#load data
#R, mask = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\R3_train.txt" , small_data=False)
#test, mask2 = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\R3_test.txt" , small_data=False)
#R, mask = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.base" , scale = True,small_data=False)
R = np.loadtxt('R_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
mask = np.loadtxt('mask_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
test, mask2 = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.test" ,scale = True ,small_data=False)
mask2 = mask2.toarray()
t = time()
print("Start Process ....")
#Q_dsgd, P_dsgd, Rmse_dsgd,T_dsgd= Parallelized_SGD(R, mask,test,mask2)
Q_sgd,P_sgd,Rmse_sgd,T_sgd=SGD2(R,mask,test,mask2)
print("Process finished in %s s"%(time()-t))
# Wrtie the obtained Matrices to csv file
outputMatrix(P_sgd,output_P)
outputMatrix(Q_sgd,output_Q)
print(time()-t)
#np.savetxt('Rmse_dsgd_normal_2.npy',Rmse_dsgd)
#np.savetxt('T_dsgd_normal_2.npy',T_dsgd)
np.savetxt('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),Rmse_sgd)
np.savetxt('T_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),T_sgd)

plt.figure()
plt.xlabel('time')
plt.ylabel('Rmse')
plt.plot(T_sgd[:10000],Rmse_sgd[:10000],'r')
plt.title('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
print('Rmse_sgd_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it))
#plt.plot(T_dsgd,Rmse_dsgd,'b')
#plt.legend(labels=["SGD","DSGD"])
