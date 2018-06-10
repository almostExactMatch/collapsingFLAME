import numpy as np
import pandas as pd
import pickle
import time
import itertools
from joblib import Parallel, delayed
from random import randint
import matplotlib
import sklearn
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

#load collapsing result    
with open('col_result1_0.20','rb') as f:
    data = pickle.load(f)    
res1 = data[1]  

effect = {}
for level in res1:
    count = set()
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect[idx] = [CATE] 
            count.add(idx)

with open('col_result2_0.20','rb') as f:
    data = pickle.load(f)    
res2 = data[1]

for level in res2:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)
            
with open('col_result3_0.20','rb') as f:
    data = pickle.load(f)    
res3 = data[1]

for level in res3:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)            
            
with open('col_result4_0.20','rb') as f:
    data = pickle.load(f)    
res4 = data[1]

for level in res4:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)     
            
with open('col_result5_0.20','rb') as f:
    data = pickle.load(f)    
res5 = data[1]  

for level in res5:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)            
            
with open('col_result6_0.20','rb') as f:
    data = pickle.load(f)    
res6 = data[1] 

for level in res6:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)            
            
with open('col_result7_0.20','rb') as f:
    data = pickle.load(f)    
res7 = data[1] 

for level in res7:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)     
                
with open('col_result8_0.20','rb') as f:
    data = pickle.load(f)    
res8 = data[1] 

for level in res8:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)    
                
with open('col_result9_0.20','rb') as f:
    data = pickle.load(f)    
res9 = data[1]

for level in res9:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)     
                
with open('col_result10_0.20','rb') as f:
    data = pickle.load(f)    
res10 = data[1] 

for level in res10:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            if idx not in effect:
                effect[idx] = [CATE] 
            else:
                effect[idx].append(CATE)   
                
for key in effect:
    effect[key] = np.mean(effect[key])

with open('col_result_0.20','rb') as f:
    data = pickle.load(f)    
res = data[1] 

effect_compare = {}
for level in res:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect_compare[idx] = CATE  
            
print(len(effect))
print(len(effect_compare))

with open('data_0.20', 'rb') as f:
    data = pickle.load(f)
miss = []
covs = [0,1,2,3,4,5,6,7,8,9]
for idx, row in data.iterrows():
    for cov in covs:
        if row[cov] == 2:
            miss.append(int(row['index0'])) 
            break
miss = set(miss)

x = [] 
y = []
count = 0.0
for key in effect:
    if key not in effect_compare:
        continue
    x.append(effect[key])
    y.append(effect_compare[key])
    if effect[key] != effect_compare[key]:
        if key in miss:
            count += 1

plt.scatter(x,y,color = 'r')
#x_ = range(-5, 30)
plt.xlim(-20,40)
plt.ylim(-20,40)
plt.title("Collapsing FLAME with 20% missing data")
#plt.plot(x_,x_, color = 'b')
plt.xlabel("Estimated CATE with multiple imputations")
plt.ylabel("Estimated CATE without imputation")
plt.savefig("col_impute_0.20.png")

print(sklearn.metrics.mean_squared_error(x, y))

col_sum = 0
gen_sum = 0
for i in range(len(x)):
    col_sum += abs(x[i] - y[i])
print(col_sum* Decimal(1.0) / Decimal(len(x)))
