
# coding: utf-8

# In[46]:

import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed
from decimal import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from operator import itemgetter
from random import randint
import operator
from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from sqlalchemy import create_engine

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import math
from sklearn.utils import shuffle

#generate_miss()
with open('data', 'rb') as f:
    data = pickle.load(f)
miss = []
covs = [0,1,2,3,4,5,6,7,8,9]
for idx, row in data.iterrows():
    for cov in covs:
        if row[cov] != 0 and row[cov] != 1:
            miss.append(int(row['index0'])) 
            
# function to compute the balancing factor
def balancing_factor(df, match_indicator, tradeoff = 0.0):
    ''' Input : 
            df : the data matrix
            match_indicator : the matched indicator column 
        
        Output : 
            balancing_factor : the balancing factor
            time_BF : time to compute the balancing factor
    '''    
    
    s = time.time()
    
    # how many control units are unmatched 
    # recall matched units are removed from the data frame
    num_control = len(df[df['treated']==0]) 
    
    # how many treated units that are unmatched
    # recall matched units are removed from the data frame
    num_treated = len(df[df['treated']==1]) 
    
    # how many control units are matched at this level
    num_control_matched = np.sum(( match_indicator ) & (df['treated']==0) )
    
    # how many treated units are matched at this level
    num_treated_matched = np.sum(( match_indicator ) & (df['treated']==1) ) 
    
    BF = tradeoff * ( float(num_control_matched)/num_control + float(num_treated_matched)/num_treated ) 
    
    time_BF = time.time() - s
    
    
    # -- below is the level-wise MQ
    return  (BF , time_BF )

# function for computing the prediction error
def prediction_error(holdout, covs_subset, ridge_reg = 0.1):
    ''' Input : 
            holdout : the training data matrix
            covs_subsets : the list of covariates to matched on
        
        Output : 
            pe : the prediction error
            time_PE : time to compute the regression
    '''    
        
   
    # below is the regression part for PE
    s = time.time()
    
    # Ridge : solves a regression model where the loss function is the linear
    #         least squares function and regularization is given by the l2-norm
    ridge_c = Ridge(alpha=ridge_reg) 
    ridge_t = Ridge(alpha=ridge_reg) 
    
    #tree_c = DecisionTreeRegressor(max_depth=8, random_state=0)
    #tree_t = DecisionTreeRegressor(max_depth=8, random_state=0)
        
    n_mse_t = np.mean(cross_val_score(ridge_t, holdout[holdout['treated']==1][covs_subset], 
                                holdout[holdout['treated']==1]['outcome'] , 
                                        scoring = 'neg_mean_squared_error' ) )
        
    n_mse_c = np.mean(cross_val_score(ridge_c, holdout[holdout['treated']==0][covs_subset], 
                                holdout[holdout['treated']==0]['outcome'] ,
                                        scoring = 'neg_mean_squared_error' ) )
    
    #n_mse_t = np.mean(cross_val_score(tree_t, holdout[holdout['treated']==1][covs_subset], 
    #                            holdout[holdout['treated']==1]['outcome'] , 
    #                                    scoring = 'neg_mean_squared_error' ) )
        
    #n_mse_c = np.mean(cross_val_score(tree_c, holdout[holdout['treated']==0][covs_subset], 
    #                            holdout[holdout['treated']==0]['outcome'] , 
    #                                    scoring = 'neg_mean_squared_error' ) )
    PE = n_mse_t + n_mse_c
    
    time_PE = time.time() - s
    # -- above is the regression part for PE
    
    # -- below is the level-wise MQ
    return  (PE, time_PE, n_mse_t, n_mse_c) 

def match(df_, covs, covs_max_list, treatment_indicator_col = 'treated', match_indicator_col = 'matched'):
    num_units = df_.shape[0]
    count = 0
    idx_miss = []
    for idx,row in df_.iterrows():
        for cov in covs:
            if row[cov] == 2: 
                idx_miss.append(count)
                break
        count += 1  
    idx_miss = np.array(idx_miss)  
    df = df_.copy()
    if len(idx_miss) != 0:
        df.drop(df.index[idx_miss], inplace=True)
    # this function takes a dataframe, a set of covariates to match on, 
    # the treatment indicator column and the matched indicator column.
    # it returns the array indicating whether each unit is matched (the first return value), 
    # and a list of indices for the matched units (the second return value)
    #print(covs)
    arr_slice_wo_t = df[covs].values # the covariates values as a matrix
    arr_slice_w_t = df[ covs + [treatment_indicator_col] ].values # the covariate values together with the treatment indicator as a matrix
        
    lidx_wo_t = np.dot( arr_slice_wo_t, np.array([ covs_max_list[i]**(len(covs_max_list) - 1 - i) for i in range(len(covs_max_list))]) ) # matrix multiplication, get a unique number for each unit
    lidx_w_t = np.dot( arr_slice_w_t, np.array([ covs_max_list[i]**(len(covs_max_list) - i) for i in range(len(covs_max_list))] +                                               [1]
                                              ) ) # matrix multiplication, get a unique number for each unit with treatment indicator
        
    _, unqtags_wo_t, counts_wo_t = np.unique(lidx_wo_t, return_inverse=True, return_counts=True) # count how many times each number appears
    _, unqtags_w_t, counts_w_t = np.unique(lidx_w_t, return_inverse=True, return_counts=True) # count how many times each number appears (with treatment indicator)
    
    match_indicator_tmp = ~(counts_w_t[unqtags_w_t] == counts_wo_t[unqtags_wo_t]) # a unit is matched if and only if the counts don't agree
    
    match_indicator = [False] * num_units
    idx_miss = set(idx_miss)
    base = 0
    for i in range(num_units):
        if i in idx_miss:
            continue
        match_indicator[i] = match_indicator_tmp[base]
        base += 1
    match_indicator = np.array(match_indicator)    
    
    return match_indicator, lidx_wo_t[match_indicator_tmp]

# match_quality, the larger the better
def match_quality(BF, PE):
    ''' Input : 
            df : the data matrix
            holdout : the training data matrix
            covs_subsets : the list of covariates to matched on
            match_indicator : the matched indicator column 
        
        Output : 
            match_quality : the matched quality
            time_PE : time to compute the regression
            time_BF : time to compute the balancing factor
    '''    
    res = BF + PE
    
    return  res

def get_CATE_bit(df, match_indicator, index):
    d = df[ match_indicator ]
    # when index == None, nothing is matched
    if index is None: 
        return None
    
    # we do a groupby to get the statistics

    d.loc[:,'grp_id'] = index
    
    res = d.groupby(['grp_id'])
    res_list = []
    count = 0
    count_miss = 0
    idx_miss = []    
    for key, item in res:
        df = res.get_group(key)
        index_list = df['index0'].tolist()
        df_t = df[df['treated']==1]
        df_c = df[df['treated']==0]
        mean_c = df_c['outcome'].mean()
        mean_t = df_t['outcome'].mean()
        mean = mean_t - mean_c
        
        if math.isnan(mean):
            print("group:")
            print(df)
            
        res_list.append([Decimal(mean),index_list])
        count += len(index_list)
        
        for idx in index_list:
            if idx in miss:
                count_miss += 1
                idx_miss.append((df,idx))        
        
    #print(count_miss)
    return res_list

def run_bit(df, holdout, covs, covs_max_list, threshold,tradeoff_param = 0.0):
    constant_list = ['outcome', 'treated','matched','index0']
    
    covs_dropped = []
    cur_covs = covs[:]
    cur_covs_max_list = covs_max_list[:]

    timings = [0]*3 # first entry - match (matrix multiplication and value counting and comparison), 
                    # second entry - keep track of CATE,
                    # third entry - update dataframe (remove matched units)
    
    level = 0
    s = time.time()
    match_indicator, index = match(df, cur_covs, covs_max_list) # match without dropping anything
    timings[0] = timings[0] + time.time() - s
    nb_match_units = [len(df[match_indicator])]
    BFs, time_BFs = balancing_factor(df, match_indicator, tradeoff=tradeoff_param)
    balance = [BFs]
    PEs, time_PE, n_mse_T, n_mse_C = prediction_error(holdout, cur_covs)
    prediction = [PEs]
    prediction_pos = [0]
    n_mse_treatment = [n_mse_T]
    n_mse_control = [n_mse_C]
    level_scores = [PEs + BFs]
    
    s = time.time()
    res = get_CATE_bit(df, match_indicator, index) # get the CATEs without dropping anything
    timings[1] = timings[1] + time.time() - s
    
    matching_res = [res] # result on first level, None says nothing is dropped
    
    s = time.time()
    df = df[~match_indicator][ cur_covs + constant_list ] # remove matched units
    timings[2] = timings[2] + time.time() - s  
    
    nb_steps = 0
    drops = [[]]
    all_covs = cur_covs
    nb_units = [len(df)]
    cov_dropped = []
    
    while len(cur_covs)>1:
        nb_steps = nb_steps + 1
        
        level += 1
        matching_result_tmp = []
        
        # the early stopping condition
        if (df['treated'] == 0).empty  | (df['treated'] == 1).empty :
            print 'no more matches'
            break
        
        if (np.sum(df['treated'] == 0)==0) | (np.sum(df['treated'] == 1)==0) :
            print 'no more matches'
            break
        
        # added to put theshold on number of units matched
        for i in range(len(cur_covs)):
            
            cur_covs_no_c = cur_covs[:i] + cur_covs[i+1:]
            
            cur_covs_max_list_no_c = cur_covs_max_list[:i] + cur_covs_max_list[i+1:]
            
            s = time.time()
            match_indicator, index = match(df, cur_covs_no_c, cur_covs_max_list_no_c)
            timings[0] = timings[0] + time.time() - s 
            
            balancing_f, time_BF = balancing_factor(df, match_indicator, tradeoff=tradeoff_param)
            prediction_e, time_PE, n_mse_t, n_mse_c = prediction_error(holdout, cur_covs_no_c)
            
            score = match_quality(balancing_f, prediction_e)
                 
            matching_result_tmp.append( (cur_covs_no_c, cur_covs_max_list_no_c, score, match_indicator, index) )
            
        best_res = max(matching_result_tmp, key=itemgetter(2)) # use the one with largest MQ as the one to drop

        level_scores.append(best_res[2])
        nb_match_units.append(len(df[best_res[-2]]))

        del matching_result_tmp
        
        s = time.time()
        new_matching_res = get_CATE_bit(df, best_res[-2], best_res[-1])
        timings[1] = timings[1] + time.time() - s     
   
        cur_covs = best_res[0] 
        cur_covs_max_list = best_res[1]
        cov_dropped = sorted(set(all_covs) - set(cur_covs) )
        drops.append(cov_dropped)
        print(cov_dropped)
        matching_res.append(new_matching_res)
        
        s = time.time()
        df = df[~ best_res[-2] ]
        timings[2] = timings[2] + time.time() - s    
        
        units_left = len(df)
        if units_left <= threshold: 
            print('reached threshold')  
            break   
        
    return (timings, matching_res, level_scores)


def run(read, write):
    with open(read,'rb') as f:
        data = pickle.load(f)

    df = data
    holdout = df
    
    res_col = run_bit(df, holdout,range(10), [2]*10, 0, tradeoff_param = 0.0) 
    pickle.dump(res_col, open(write, 'wb'))


reads = ['data','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10']
writes = ['gen_result_0.20','gen_result1_0.20','gen_result2_0.20','gen_result3_0.20','gen_result4_0.20','gen_result5_0.20','gen_result6_0.20','gen_result7_0.20','gen_result8_0.20','gen_result9_0.20','gen_result10_0.20']
count = 0
for i in range(11):
    print(count)
    count += 1
    run(reads[i],writes[i]) 
'''

run('full','gen_result_full')    
'''