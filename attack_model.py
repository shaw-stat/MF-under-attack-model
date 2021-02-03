# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:37:03 2021

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

from dsgd import *

def random_attack(R_ord, at_size, fil_size, it):
    R = R_ord
    L = R.shape[0]
    M = R.shape[1]
    mean_r = R[R.nonzero()].mean()
    std_r = R[R.nonzero()].std()
    max_r = R[R.nonzero()].max()
    at_range = random.sample(range(0,L-1),int(L*at_size))
    for i in at_range:
        r_range = random.sample(range(0,M-1),int(M*fil_size))
        R[i,r_range] = np.random.normal(mean_r,std_r,len(r_range))
        R[i,it] = max_r
    
    return R

def avg_attack(R_ord, at_size, fil_size, it):
    R = R_ord
    L = R.shape[0]
    M = R.shape[1]
    max_r = R[R.nonzero()].max()
    at_range = random.sample(range(0,L-1),int(L*at_size))
    for i in at_range:
        r_range = random.sample(range(0,M-1),int(M*fil_size))
        mean_r = R[i,:][R[i,:].nonzero()].mean()
        std_r = R[i,:][R[i,:].nonzero()].std()
        R[i,r_range] = np.random.normal(mean_r,std_r,len(r_range))
        R[i,it] = max_r
    
    return R


if __name__=="__main__":
    R, mask = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.base" , scale = True,small_data=False)
    test, mask2 = load_data(filename="D:\\新建文件夹2019\\SG_MCMC\\ua.test" ,scale = True ,small_data=False)
    at_size = 0.3
    fil_size = 0.5
    it = 1
    #R_atk = random_attack(R, at_size, fil_size, it)
    R_atk = avg_attack(R, at_size, fil_size, it)
    mask_atk =R_atk==0
    np.savetxt('mask_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),mask_atk)
    np.savetxt('R_atk_avg_%f_%f_%d.npy'%(at_size,fil_size,it),R_atk)
