# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed

import matplotlib
matplotlib.rcParams.update({'font.size': 17.5})

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
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



# In[2]:

def construct_sec_order(arr):
    ''' an intermediate data generation function used 
        for generating second order information '''
    
    second_order_feature = []
    num_cov_sec = len(arr[0])
    for a in arr:
        tmp = []
        for i in range(num_cov_sec):
            for j in range(i+1, num_cov_sec):
                tmp.append( a[i] * a[j] )
        second_order_feature.append(tmp)
        
    return np.array(second_order_feature)




# In[3]:

def data_generation_dense_2(num_control, num_treated, num_cov_dense, 
                            num_covs_unimportant, control_m = 0.1,
                            treated_m = 0.9):
    
    ''' the data generating function that we will use; 
        includes second order information '''
    

    # generate data for control group 
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense)) 
    
    # generate data for treated group 
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   
     
    
    errors1 = np.random.normal(0, 0.1, size=num_control)    # some noise
    
    errors2 = np.random.normal(0, 0.1, size=num_treated)    # some noise
    
    dense_bs_sign = np.random.choice([-1,1], num_cov_dense) 
    
    dense_bs = [ np.random.normal(s * 10, 1) for s in dense_bs_sign ]

    # y for control group 
    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     
       
    # y for treated group 
    treatment_eff_coef = np.random.normal( 1.5, 0.15, size=num_cov_dense)
    treatment_effect = np.dot(xt, treatment_eff_coef) 
    
    second = construct_sec_order(xt[:,:5])
    treatment_eff_sec = np.sum(second, axis=1)
    
    yt = np.dot(xt,np.array(dense_bs))+treatment_effect+treatment_eff_sec 
                                      # + errors2    

    # generate unimportant covariates for control group
    xc2 = np.random.binomial(1, control_m, size=(num_control,
                                                 num_covs_unimportant))  
    
    # generate unimportant covariates for treated group
    xt2 = np.random.binomial(1, treated_m, size=(num_treated,
                                                 num_covs_unimportant))   
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns=range(num_cov_dense + num_covs_unimportant))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns=range(num_cov_dense + num_covs_unimportant)) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef



# In[4]:

def match_mp(df, covs, covs_max_list, 
          treatment_indicator_col='treated', match_indicator_col='matched'):
    
    ''' Input: 
            df : a dataframe,
            covs : a set of covariates to match on, 
            covs_max_list : 
            treatment_indicator_col : the treatment indicator column
            match_indicator : the matched indicator column.
        Output : 
            match_indicator : array indicating whether each unit is matched
            indices :  a list of indices for the matched units
    '''

    # truncate the matrix with the covariates columns
    arr_slice_wo_t = df[covs].values # the covariates values as a matrix
    
    # truncate the matrix with the covariate and treatment indicator columns
    arr_slice_w_t = df[ covs + [treatment_indicator_col] ].values 
    
    # matrix multiplication: get a unique number for each unit
    lidx_wo_t = np.dot( arr_slice_wo_t, 
                      np.array([covs_max_list[i]**(len(covs_max_list)-1-i)
                                   for i in range(len(covs_max_list))] 
                                ) ) 
    
    # get a unique number for each unit with treatment indicator
    lidx_w_t = np.dot( arr_slice_w_t, 
                       np.array([covs_max_list[i]**(len(covs_max_list)-i) 
                                 for i in range(len(covs_max_list))] + [1]
                               ) ) 
    
    # count how many times each number appears
    _, unqtags_wo_t, counts_wo_t = np.unique(lidx_wo_t, return_inverse=True,
                                                        return_counts=True)
                                                        
    # count how many times each number appears (with treatment indicator)
    _, unqtags_w_t, counts_w_t = np.unique(lidx_w_t, return_inverse=True, 
                                                     return_counts=True) 
    
    # a unit is matched if and only if the counts don't agree
    match_indicator = ~(counts_w_t[unqtags_w_t] == counts_wo_t[unqtags_wo_t]) 
        
    return match_indicator, lidx_wo_t[match_indicator]



# In[5]:

# function for computing the prediction error
def prediction_error_mp(holdout, covs_subset, ridge_reg = 0.1):
    ''' Input : 
            holdout : the training data matrix
            covs_subsets : the list of covariates to matched on
        
        Output : 
            pe : the prediction error
            time_PE : time to compute the regression
    '''    
        
   
    # below is the regression part for PE
    s = time.time()
    
    # Ridge : solves a regression model where the loss function is 
    #         the linear least squares function and 
    #         regularization is given by the l2-norm
    ridge_c = Ridge(alpha=ridge_reg) 
    ridge_t = Ridge(alpha=ridge_reg) 
    
       
    n_mse_t = np.mean(cross_val_score(ridge_t,
                                holdout[holdout['treated']==1][covs_subset], 
                                holdout[holdout['treated']==1]['outcome'], 
                                scoring = 'neg_mean_squared_error' ) )
        
    n_mse_c = np.mean(cross_val_score(ridge_c, 
                                holdout[holdout['treated']==0][covs_subset], 
                                holdout[holdout['treated']==0]['outcome'],
                                scoring = 'neg_mean_squared_error' ) )
    

    PE = n_mse_t + n_mse_c
    
    time_PE = time.time() - s
    # -- above is the regression part for PE
    
    # -- below is the level-wise MQ
    return  (PE, time_PE,  n_mse_t, n_mse_c) 
    
    


# In[6]:

# function to compute the balancing factor
def balancing_factor_mp(df, match_indicator, tradeoff = 0.1):
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
    num_control_matched = np.sum((match_indicator) & (df['treated']==0))
    
    # how many treated units are matched at this level
    num_treated_matched = np.sum((match_indicator) & (df['treated']==1)) 
    
    BF = tradeoff * ( float(num_control_matched)/num_control + 
                      float(num_treated_matched)/num_treated ) 
    
    time_BF = time.time() - s
    
    
    # -- below is the level-wise MQ
    return  (BF , time_BF ) 
    
    


# In[7]:

# match_quality, the larger the better
def match_quality_mp(BF, PE):
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
    
    
    return  (BF + PE) 
    
   


# In[8]:

def get_CATE_bit_mp(df, match_indicator, index):
    d = df[ match_indicator ]
    
    # when index == None, nothing is matched
    if index is None: 
        return None
    
    # we do a groupby to get the statistics
    d.loc[:,'grp_id'] = index
    res = d.groupby(['grp_id', 'treated'])['outcome'].aggregate([np.size,
                                                                 np.mean]) 
    
    return res


# In[9]:

def recover_covs_mp(d, covs, covs_max_list, binary = True):
        
    ind = d.index.get_level_values(0)
    ind = [ num2vec_mp(ind[i], covs_max_list) 
           for i in range(len(ind)) if i%2==0]

    df = pd.DataFrame(ind, columns=covs ).astype(int)



    mean_list = list(d['mean'])
    size_list = list(d['size'])
        
    effect_list = [mean_list[2*i+1] - mean_list[2*i] 
                   for i in range(len(mean_list)//2) ]
    df.loc[:,'effect'] = effect_list
    df.loc[:,'size'] = [size_list[2*i+1] + size_list[2*i] 
                        for i in range(len(size_list)//2) ]
    
    return df



# In[10]:

def cleanup_result_mp(res_all):
    res = []
    for i in range(len(res_all)):
        r = res_all[i]
        if not r[1] is None:
            res.append(recover_covs_mp( r[1], r[0][0], r[0][1] ) )
    return res




# In[11]:

def num2vec_mp(num, covs_max_list):
    res = []
    for i in range(len(covs_max_list)):
        num_i = num/covs_max_list[i]**(len(covs_max_list)-1-i)
        res.append(num_i)
        
        if (num_i == 0) & (num%covs_max_list[i]**(len(covs_max_list)-1-i) == 0):
            res = res + [0]*(len(covs_max_list)-1-i)
            break
        num = num - num_i* covs_max_list[i]**(len(covs_max_list)-1-i)
    return res



# In[12]:

from itertools import combinations
import re

class PredictionE_mp: # Create a class for the prediction error of each set: 
                   # PE^k = {PE(s) | len(s) = k}
    """Class to define the set of Prediction Error for sets of size k : 
       PE^k characterized by:
    - k = size of the sets
    - sets: pred_e : a set and the corresponding prediction error
    """


    # We can create the PE^k by specifying k=size
    # In the context of FLAME, all the PE^k are initialized by:
    #     min_support = k, k=1..n with n = number of covariates
    #     sets = []
    #     pred_E = [0]*n since we have n covariates
    
    def __init__(self, size, sets, cur_set, pred_e):
        self.size = size
        self.sets = {cur_set : pred_e}
    
    def add(self, new_set, new_pred_error):
        """ this method adds the new set to the sets and 
            the corresponding prediction error"""
        
        self.sets[new_set] = new_pred_error
        
       


# In[13]:

from itertools import combinations
import re


class DroppedSets_mp: # Create a class for the dropped sets : 
                   # D^k = {s | s has been dropped and len(s) = k}
    """Class to define the set of dropped sets of size k : 
       D^k characterized by:
    - min_support : the size of the itemsets in the set 
    - dropped : set of the dropped sets
    - support : list of the current support of each item in the dropped set
    - min_support_items : set of items that have minimum support """


    # We can create the D^k by specifying k=min_support, 
    # In the context of FLAME, all the D^k are initialized by:
    #     min_support = k, k=1..n with n = number of covariates
    #     dropped = []
    #     support = [0]*n since we have n covariates
    #     min_support_items = []


    
    def __init__(self, min_sup, dropped, support, min_sup_item):
        self.min_support = min_sup
        self.dropped = dropped
        self.support = support
        self.min_support_item = min_sup_item
    
    def add(self, new_set):
        """ this method adds the new set to the dropped set and 
            updates the support for the current items and 
            the items with enough support"""
        
        # update the set of dropped sets
        self.dropped.append(new_set)
        self.dropped = sorted(self.dropped)
        
        # update the support of the items in the new_set
        for item in new_set:
            self.support[item] += 1
            
            # update the list of items with enough support
            if self.support[item] >= self.min_support:
                self.min_support_item.append(item)
        self.min_support_item = sorted(self.min_support_item)
    
    def generate_active_sets(self, new_set):
        """ this method generates the new active sets from 
            the current dropped set"""
        
        new_active_sets = []
        new_candidate = []
        rem = []
        
        # start by verifying if all the items in new_set have min support : 
        #     if no, there is no new active set to generate
        #     if yes, create a new active set by joining the set 
        #     with the items of min support
        
        if set(new_set).issubset(set(self.min_support_item)) :
            aux = sorted(set(self.min_support_item) - set(new_set))
            
            for element in aux:
                new_candidate = sorted(set(new_set).union(set([element])))
                new_active_sets.append(new_candidate)
            
        # now we can test if each candidate can be dropped
        for c in new_active_sets:
            
            # generates the subsets needed to have already been dropped
            # the ouptut of combinations is a list of tuples
            prefix = combinations(c,self.min_support) 
        
            for c_p in set(prefix):
                if sorted(c_p) not in self.dropped : 
                    # if a prefix of 'c' has not been dropped yet,
                    # remove 'c' from candidates
                    rem.append(c)
                    break # no need to check if the others 
                          # prefixes have been dropped
        for r in rem:
            new_active_sets.remove(r)
            # new_active_sets contains the sets to add to possible_drops
        
        return new_active_sets


# In[14]:

def run_mpbit(df, holdout, covs, covs_max_list, covs_unimportant,threshold, tradeoff_param = 0.1):
    
    
    #----------- INITIALIZE THE DIFFERENT PARAMETERS ---------------#
    
    constant_list = ['outcome', 'treated']
    
    covs_dropped = [] # set of sets of covariates dropped
    all_covs = covs[:] # set of all covariates
    
    cur_covs_max_list = covs_max_list[:]
    pos_drops = [[covs[i]] for i in range(len(covs))]
    
    drops = [[]] # to keep track of the sets dropped
    len_queue = [len(pos_drops)]
    
    # initialize the sets of dropped sets of size k, k=1..num_covs
    # D^k = {s | s has been dropped and len(s) = k }
    # we use the DroppedSets class
    num_covs = len(covs)
    D = []
    for k in range(1,num_covs+1): 
        D.append(DroppedSets_mp(k, [], [0]*num_covs, [])) 
        # D[k] is for the dropped sets of size k+1
    

    # initialize the PE for sets of size k, k=1..num_covs
    # PE^k
    # we use the PredictionE class
    
    PE = []  #PE[k] contains the PE for dropped sets of size k
    
    
    for k in range(1, num_covs+1): 
        PE.append(PredictionE_mp(k, {}, (), 0)) 
    
    # initialize the timing table
    t_match, t_pe, t_bf, t_cate, t_rm, = ([],[0],[0],[], []) 
    
    #--------- MATCH WITHOUT DROPPING ANYTHING AND GET CATE ----------#

    nb_steps = 0
    
    # match without dropping anything
    t = time.time()
    match_indicator, index = match_mp(df, all_covs, covs_max_list) 
    t_match.append(time.time() - t)
    
    nb_match_units = [len(df[match_indicator])]
    BFs, time_BFs = balancing_factor_mp(df, match_indicator,
                                     tradeoff=tradeoff_param)
    balance = [BFs]
    PEs, time_PE, n_mse_T, n_mse_C = prediction_error_mp(holdout, covs)
    prediction = [PEs]
    prediction_pos = [0]
    n_mse_treatment = [n_mse_T]
    n_mse_control = [n_mse_C]
    level_scores = [PEs + BFs]

    nb_units_t = [len(df[df['treated'] == 1])]
    nb_units_c = [len(df[df['treated'] == 0])]
    # get the CATEs without dropping anything
    t = time.time()
    res = get_CATE_bit_mp(df, match_indicator, index) 
    t_cate.append(time.time() - t)

    
    # result on first level, None means nothing is dropped
    matching_res = [[( all_covs, cur_covs_max_list, None,
                      match_indicator, index), res]] 
    
    # remove matched units
    t = time.time()
    df = df[~match_indicator][ all_covs + constant_list ] 
    t_rm.append(time.time() - t)




    #-------------------- RUN COLLAPSING FLAME  ----------------------#


    while len(pos_drops)>0: # we still have sets to drop
        
        nb_steps = nb_steps + 1
        
        # new stoping criteria
        if pos_drops == [all_covs]: 
            print('all possibles sets dropped')  
            break
        
        
        best_score = np.inf
        matching_result_tmp = []

        # early stopping condition
        if (df['treated'] == 0).empty  | (df['treated'] == 1).empty: 
            print('no more matches')
            break
            
        if (np.sum(df['treated'] == 0)==0) | (np.sum(df['treated'] == 1)==0) : 
            print('no more matches')
            break
        
        # added to put theshold on number of units matched
        units_left = len(df)
        #print units_left
        if units_left <= threshold: 
            print('reached threshold')  
            break
        
        t_match_tmp = 0
        t_pe_tmp = 0
        t_bf_tmp = 0
        
        # PROVE that we only need to look at the largest sets in the queue
        #size = []
        #for i in pos_drops:
        #    size.append(len(i))
            
        #max_size = max(size)
        #to_drop = []
        #for x in pos_drops:
        #    if len(x) == max_size:
        #        to_drop.append(x)
        
        #------------------ FIND THE SET TO DROP ----------------------#
        for s in pos_drops:
            
            cur_covs_no_s = sorted(set(all_covs) - set(s))
            cur_covs_max_list_no_s = [2]*(len(all_covs) - len(s))

            #### To do : compute the error and if err + c*2 
            #### is worse than all the other then,
            #### we don't need to do the Exact Match
            t = time.time()
            match_indicator, index = match_mp(df, cur_covs_no_s,
                                           cur_covs_max_list_no_s) 
            t_match_tmp = t_match_tmp + time.time() - t 

            BF, time_BF = balancing_factor_mp(df, match_indicator,
                                           tradeoff=tradeoff_param)
            t_bf_tmp = t_bf_tmp + time_BF


            if tuple(s) not in PE[len(s)].sets.keys():
                tmp_pe, time_PE, n_mse_t, n_mse_c = prediction_error_mp(holdout,
                                                                     cur_covs_no_s)
                PE[len(s)].sets[tuple(s)] = tmp_pe
            
            pe_s = PE[len(s)].sets[tuple(s)] 
            prediction_pos.append(pe_s)
            t_pe_tmp = t_pe_tmp + time_PE 


            score = match_quality_mp(BF, pe_s)


                         
            matching_result_tmp.append((cur_covs_no_s, cur_covs_max_list_no_s,
                                         score, match_indicator, index) )
            
        #-------------------- SET TO DROP FOUND ------------------------#


        #------------------- UPDATE THE TIMING TABLE -------------------#


        t_match.append(t_match_tmp)
        t_pe.append(t_pe_tmp)
        t_bf.append(t_bf_tmp)
        del(t_match_tmp)
        del(t_pe_tmp)
        del(t_bf_tmp)
        #del(to_drop)
        
        #------------------- TIMING TABLE UPDATED ---------------------#


        #------- DROP THE SET AND UPDATE MATCHING QUALITY AND CATE  ---#
        
        # choose the set with largest MQ as the set to drop
        best_res = max(matching_result_tmp, key=itemgetter(2)) 
        # balanced decrease treshold 
#        if best_res[2] < threshold: 
#            print('reached threshold')  
#            break
            
        level_scores.append(max( [t[2] for t in matching_result_tmp] )) # just take best_res[2]
        nb_match_units.append(len(df[best_res[-2]]))
         
        del(matching_result_tmp)
        
        t = time.time()
        new_matching_res = get_CATE_bit_mp(df, best_res[-2], best_res[-1])
        t_cate.append(time.time() - t)

        
        covs_used = best_res[0]
        cur_covs_max_list = best_res[1]
        matching_res.append([best_res, new_matching_res])
        
        set_dropped = sorted(set(all_covs) - set(covs_used))
        
        #to have the PE and BF and each level
        cur_covs_no_s = sorted(set(covs_used))
        cur_covs_max_list_no_s = [2]*(len(covs_used))
            

        
        #---- SET DROPPED AND MATCHING QUALITY AND CATE UPDATED ------#


        #------- GENERATE NEW ACTIVE SETS AND UPDATE THE QUEUE -------#


        #new steps to find the new set of possible drops
        
        drops.append(set_dropped) # to keep track of the dropped sets
        pos_drops = sorted(pos_drops)
        
        # remove the dropped set from the set of possible drops
        pos_drops.remove(set_dropped)
        
        # add the dropped set to the set of dropped covariates
        covs_dropped.append(set_dropped) 
        
        # add set_dropped to the correct D^k
        k = len(set_dropped)
        D[k-1].add(set_dropped)
        
       
        # now generate the new active sets from set_dropped
        new_active_drops = D[k-1].generate_active_sets(set_dropped)
        
        # add new_active_drops to possible drops
        for x in new_active_drops: 
            if x not in pos_drops:
                pos_drops.append(x) 
        
        len_queue.append(len(pos_drops))
        
        
        t = time.time()
        df = df[~ best_res[-2] ]

        t_rm.append(time.time() - t)
        
        #------------------- QUEUE UPDATED -----------------------------#
        #early stopping : before dropping important covs
        if len(set_dropped) == covs_unimportant: 
            print('all important covariate dropped')  
            break
                
        

    #---------- END COLLAPSING FLAME : RETURN RESULTS ------------------#

    
    timings = (t_match, t_pe, t_bf, t_cate, t_rm)
    
    return (timings, cleanup_result_mp(matching_res), nb_steps,
            level_scores, drops, len_queue, nb_match_units,
            balance, prediction, n_mse_treatment, n_mse_control, nb_units_c, nb_units_t)


# In[15]: