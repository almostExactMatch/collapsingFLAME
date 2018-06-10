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
with open('col_result1','rb') as f:
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

with open('col_result2','rb') as f:
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
            
with open('col_result3','rb') as f:
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
            
with open('col_result4','rb') as f:
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
            
with open('col_result5','rb') as f:
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
            
with open('col_result6','rb') as f:
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
            
with open('col_result7','rb') as f:
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
                
with open('col_result8','rb') as f:
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
                
with open('col_result9','rb') as f:
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
                
with open('col_result10','rb') as f:
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

with open('col_result','rb') as f:
    data = pickle.load(f)    
res = data[1] 

effect_compare = {}
for level in res:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect_compare[idx] = CATE  
            

with open('data', 'rb') as f:
    data = pickle.load(f)
miss = []
covs = [0,1,2,3,4,5,6,7,8,9]
for idx, row in data.iterrows():
    for cov in covs:
        if row[cov] == 2:
            miss.append(int(row['index0'])) 
            break
miss = set(miss)

x_1 = [] 
y_1 = []
count = 0.0
for key in effect:
    if key not in effect_compare:
        continue
    x_1.append(effect[key])
    y_1.append(effect_compare[key])
    if effect[key] != effect_compare[key]:
        if key in miss:
            count += 1     
            

with open('col_result','rb') as f:
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
x_3 = []
y_3 = []
no_match = []
for key in effect_compare:
    #print(key, effect[key])
    if key not in effect_dic:
        continue
    x_3.append(effect_compare[key])
    y_3.append(effect_dic[key])
    
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
          
#load collapsing result    
with open('gen_result1','rb') as f:
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

with open('gen_result2','rb') as f:
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
            
with open('gen_result3','rb') as f:
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
            
with open('gen_result4','rb') as f:
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
            
with open('gen_result5','rb') as f:
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
            
with open('gen_result6','rb') as f:
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
            
with open('gen_result7','rb') as f:
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
                
with open('gen_result8','rb') as f:
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
                
with open('gen_result9','rb') as f:
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
                
with open('gen_result10','rb') as f:
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

with open('gen_result','rb') as f:
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

with open('data', 'rb') as f:
    data = pickle.load(f)
miss = []
covs = [0,1,2,3,4,5,6,7,8,9]
for idx, row in data.iterrows():
    for cov in covs:
        if row[cov] == 2:
            miss.append(int(row['index0'])) 
            break
miss = set(miss)
print(miss)
x_2 = [] 
y_2 = []
count = 0.0
for key in effect:
    if key not in effect_compare:
        continue
    x_2.append(effect[key])
    y_2.append(effect_compare[key])
    if effect[key] != effect_compare[key]:
        if key in miss:
            count += 1

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
x_4 = []
y_4 = []
no_match = []
for key in effect_compare:
    #print(key, effect[key])
    if key not in effect_dic:
        continue
    x_4.append(effect_compare[key])
    y_4.append(effect_dic[key])
    
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

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
ax1.subplot(1, 4, 1)
ax1.scatter(x_1,y_1,color = 'r')
ax1.xlim(-20,40)
ax1.ylim(-20,40)
ax1.title("Collapsing FLAME")
ax1.xlabel("Estimated CATE with multiple imputations")
ax1.ylabel("Estimated CATE without imputation")

ax2.subplot(1,4,2)
ax2.scatter(x_3,y_3,color='r')
ax2.xlim(-20,40)
ax2.ylim(-20,40)
ax2.xlabel("Estimated CATE without imputation")
ax2.ylabel("True CATE")
ax2.title("Collapsing FLAME")

ax3.subplot(1, 4, 3)
ax3.scatter(x_2,y_2,color = 'r')
ax3.xlim(-20,40)
ax3.ylim(-20,40)
ax3.title("Generic FLAME")
ax3.xlabel("Estimated CATE with multiple imputations")
ax3.ylabel("Estimated CATE without imputation")

ax4.subplot(1,4,4)
ax4.scatter(x_4,y_4,color='r')
ax4.xlim(-20,40)
ax4.ylim(-20,40)
ax4.xlabel("Estimated CATE without imputation")
ax4.ylabel("True CATE")
ax4.title("Generic FLAME")

plt.tight_layout()
plt.savefig("0.05.png")
