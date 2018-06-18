import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine
import psycopg2
from sklearn.utils import shuffle
import sql
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cPickle as pickle
from decimal import *

#get matching result for generic FLAME
with open('FLAME-gen-result','rb') as f:
    data = pickle.load(f)     
match_list_gen = data[0]

gen_group = {}
level = 1
for match in match_list_gen: 
    level_index = []

    if match is None or match.empty:
        gen_group[level] = level_index
        level += 1
        continue
    
    for idx, row in match.iterrows():
        index_list = row['index']
        index_list = [int(elem) for elem in index_list]
        for elem in index_list:
            level_index.append(elem)
        mean = row['mean']
    gen_group[level] = level_index      
    level += 1
    
#get matching result for collapsing FLAME    
with open('FLAME-col-result','rb') as f:
    data = pickle.load(f) 
match_list_col = data[1]

level = 1
col_group = {}
for match in match_list_col: 
    level_index = []
    for group in match:
        index_list = group[1]
        mean = group[0]
        for elem in index_list:
            level_index.append(elem)   
    col_group[level] = level_index      
    level += 1
    
#print result
for level in range(1,5):
    print("level: ", level)
    intersect = set(gen_group[level]).intersection(set(col_group[level]))
    print("Generic matched on : ", len(gen_group[level]))
    print("Collapsing matched on : ", len(col_group[level]))
    print("Overlapping Units: ", intersect)
    print("Number of overlapping units: ", len(intersect)) 
