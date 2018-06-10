import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed
from random import randint
import matplotlib
matplotlib.rcParams.update({'font.size': 17.5})

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from operator import itemgetter

import operator
from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from sqlalchemy import create_engine

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
from decimal import *
from numpy import genfromtxt

def construct_sec_order(arr):
    # an intermediate data generation function used for generating second order information
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)


        
def data_generation_dense_2(read,write,num_control, num_treated, num_cov_dense,
                            control_m = 0.1, treated_m = 0.9):
    
    # the data generating function that we will use. include second order information
    df = pd.read_csv(read)

    covs = ['X0','X1','X2','X3','X4','X5','X6','X7','X8','X9']
    df = df[covs]
    x = df.values
    
    xc = x[:10000,:]   # data for conum_treatedrol group
    xt = x[10000:,:]   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec #+ errors2    # y for treated group 
    
    #xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    #xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   # unimportant covariates for treated group
    
    #df1 = pd.DataFrame(np.hstack([xc, xc2]), 
    #                   columns = range(num_cov_dense + num_covs_unimportant))
    df1 = pd.DataFrame(xc, columns = range(10))    
    df1['outcome'] = yc
    df1['treated'] = 0
    
    #df2 = pd.DataFrame(np.hstack([xt, xt2]), 
    #                   columns = range(num_cov_dense + num_covs_unimportant ) ) 
    df2 = pd.DataFrame(xt, columns = range(10))   
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
    df = df.reset_index()
    df['index0'] = df.index
    df = df.drop('index', 1)
    pickle.dump(df, open(write, 'wb'))
    

reads = ['Impute1.csv','Impute2.csv','Impute3.csv','Impute4.csv','Impute5.csv','Impute6.csv','Impute7.csv','Impute8.csv','Impute8.csv','Impute9.csv','Impute10.csv']
writes = ['data1','data2','data3','data4','data5','data6','data7','data8','data9','data10']
for i in range(10):
    data_generation_dense_2(reads[i],writes[i],10000, 10000, 10,control_m = 0.1, treated_m = 0.9)   