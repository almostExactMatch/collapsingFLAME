import numpy as np
import pandas as pd
import pickle
import time
import itertools
from joblib import Parallel, delayed
from random import randint
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from operator import itemgetter
import sklearn
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

with open('gen_result','rb') as f:
    data = pickle.load(f)    
res = data[1] 

effect_compare = {}
for level in res:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect_compare[idx] = CATE  

    
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
         
treatment_eff_coef = [ 1.58433585,1.62764645,1.43909794,1.56505663,1.41105645,1.37671339,
  1.75006881,1.75391308,1.43927549,1.64816271]

'''
treatment_eff_coef = [ 1.3114241,1.37475154,1.69863368,1.60038454,1.34793361,1.45734672,
  1.48787906,1.42089063,1.53237313,1.51668491]
'''

with open('full','rb') as f:
    data = pickle.load(f)
res_true = data

#res_true = pd.read_csv('full.csv')
xt = res_true[[0,1,2,3,4,5,6,7,8,9]].values
#xt = res_true[['0','1','2','3','4','5','6','7','8','9']].values
treatment_effect = np.dot(xt, treatment_eff_coef).reshape((20000,1))
second = construct_sec_order(xt[:,:5])
treatment_eff_sec = np.sum(second, axis=1).reshape((20000,1))
effect_true = (treatment_effect + treatment_eff_sec).reshape((20000,1))
res_true['effect'] = effect_true

effect_dic = {}
for idx, row in res_true.iterrows():
    effect_dic[row['index0']] = row['effect']

#plot miss vs true
x = []
y = []
no_match = []
for key in effect_compare:
    #print(key, effect[key])
    if key not in effect_dic:
        continue
    x.append(effect_compare[key])
    y.append(effect_dic[key])
    
    if effect_compare[key] != effect_dic[key]:
        no_match.append(key)

#generate_miss()
with open('data', 'rb') as f:
    data = pickle.load(f)
miss = []
covs = [0,1,2,3,4,5,6,7,8,9]
for idx, row in data.iterrows():
    for cov in covs:
        if row[cov] != 0 and row[cov] != 1:
            miss.append(int(row['index0']))

count = 0.0
for idx in no_match:
    if idx in miss:
        count += 1

plt.scatter(x,y,color='r')
plt.xlim(-20,40)
plt.ylim(-20,40)
plt.xlabel("Estimated CATE without imputation")
plt.ylabel("True CATE")
plt.title("Generic FLAME with 5% missing data")
plt.savefig("gen_true_0.05.png")

total = Decimal(0.0)
for i in range(len(x)):

    total += abs(Decimal(x[i]) - Decimal(y[i]))
#print(total, len(x), len(y))  

#print(x,y)
print(sklearn.metrics.mean_squared_error(x,y))