import numpy as np
import pandas as pd
import pickle
import time
import itertools
from joblib import Parallel, delayed
from random import randint
import matplotlib
matplotlib.rcParams.update({'font.size': 17.5})

import matplotlib.pyplot as plt
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
from scipy.stats import multivariate_normal
from scipy import random, linalg

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

def set_missing(df):
    tradeoff = 0.20
    num_select = int(df.shape[0] * tradeoff)
    num_obs = 20000
    num_cov = 10
    select_set = set()
    df_copy = df.copy()
    for m in range(num_select):
        i = np.random.randint(1,num_obs)
        j = np.random.randint(0,num_cov)
        if (i,j) not in select_set :
            df.iloc[i,j] = np.nan
            df_copy.iloc[i,j] = 2
            select_set.add((i,j))       
    return df, df_copy

def data_generation_dense_2(x,num_control, num_treated, num_cov_dense, num_covs_unimportant,
                            control_m = 0.1, treated_m = 0.9):
    
    # the data generating function that we will use. include second order information
    xc = x[:15000,:]   # data for conum_treatedrol group
    xt = x[15000:,:]   # data for treatmenum_treated group
        
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]
    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    print(treatment_eff_coef)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt, np.array(dense_bs)) + treatment_effect + treatment_eff_sec #+ errors2    # y for treated group 
    effect = (treatment_effect + treatment_eff_sec).reshape(5000,1)
    xt = np.concatenate((xt,effect), axis = 1)
    #xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    #xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   # unimportant covariates for treated group
    
    #df1 = pd.DataFrame(np.hstack([xc, xc2]), 
    #                   columns = range(num_cov_dense + num_covs_unimportant))
    df1 = pd.DataFrame(xc, columns = range(10))    
    df1['outcome'] = yc
    df1['treated'] = 0
    df1['effect'] = None
    
    #df2 = pd.DataFrame(np.hstack([xt, xt2]), 
    #                   columns = range(num_cov_dense + num_covs_unimportant ) ) 
    c = range(10)
    c.append('effect')
    df2 = pd.DataFrame(xt, columns = c)
    df2['outcome'] = yt
    df2['treated'] = 1
    
    df = pd.concat([df1,df2])
    df['matched'] = 0
    df = df.reset_index()
    df['index0'] = df.index
    df = df.drop('effect', axis = 1)
    return df

def generate_miss():
    # the data generating function that we will use. include second order information
    matrixSize = 10 
    mu = [0] * matrixSize
    base = random.rand(matrixSize,matrixSize)
    sigma = np.dot(base,base.transpose())
    pickle.dump(sigma, open('sigma_0.20', 'wb'))
    
    x_1 = []
    for i in range(15000):
        z_i = list(np.random.multivariate_normal(mu,sigma))
        #x_i_1 = [1 if elem > 0 else 0 for elem in z_i[:-2]]
        #x_i_2 = [1 if elem > 2 else 0 for elem in z_i[-2:]]
        #x_i = x_i_1 + x_i_2
        x_i = [1 if elem > 0 else 0 for elem in z_i]
        x_1.append(x_i)
    x_2 = []
    for i in range(5000):
        z_i = list(np.random.multivariate_normal(mu,sigma))
        #x_i_1 = [1 if elem > 0 else 0 for elem in z_i[:-2]]
        #x_i_2 = [1 if elem > -2 else 0 for elem in z_i[-2:]]
        #x_i = x_i_1 + x_i_2
        x_i = [1 if elem > 0 else 0 for elem in z_i]
        x_2.append(x_i) 
      
    x = x_1 + x_2    
    x = np.array(x) 
    x = data_generation_dense_2(x,15000,5000,10,0)
    x = pd.DataFrame(x)
    
    x = x.rename(columns = {"Unnamed: 0":"index"})
    x = x.drop("index",1)
    pickle.dump(x, open('full_0.20', 'wb'))
    x.to_csv('full_0.20.csv')
    
    x, x_ = set_missing(x)
    pickle.dump(x_, open('data_0.20', 'wb'))
    x.to_csv("x_0.20.csv")
    
#generate_miss()
'''
with open('data', 'rb') as f:
    data = pickle.load(f)
idx_select = [873,10013,2269,11733,409,10875,6176,10358]
data = data.loc[data['index0'].isin(idx_select)]
print(data)
'''
with open("full", "rb") as f:
    sigma = pickle.load(f)
'''    
sigma = sigma.tolist()
res = []
for elem in sigma:
    for el in elem:
        res.append(el)
'''
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.savefig("corr_matrix.png")


df = sigma
correlation_matrix(df)
