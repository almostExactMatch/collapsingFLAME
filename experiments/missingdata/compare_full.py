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
with open('result_full','rb') as f:
    data = pickle.load(f)    
res = data[1] 

effect = {}
for level in res:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect[idx] = CATE  
            
with open('result','rb') as f:
    data = pickle.load(f)    
res = data[1] 

effect_compare = {}
for level in res:
    for elem in level:
        CATE = elem[0]
        for idx in elem[1]:
            effect_compare[idx] = CATE  
                  
x = [] 
y = []
for key in effect:
    if key not in effect_compare:
        continue
    x.append(effect[key])
    y.append(effect_compare[key])

plt.scatter(x,y,color = 'r')
#x_ = range(-5, 30)
#plt.xlim(-40,60)
#plt.ylim(-40,60)
plt.plot(x,x, color = 'b')
plt.xlabel("full data")
plt.ylabel("missing data")
plt.savefig("compare_missing_full.png")
