
# coding: utf-8

# # Sanity Check Generic FLAME
# ---

# In[ ]:




# In[23]:




# In[46]:

import numpy as np
import pandas as pd
#import pyodbc
import pickle
import time
import itertools
from joblib import Parallel, delayed

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




# In[24]:


# In[49]:

def data_generation_gradual_decrease_imbalance(num_control, num_treated, num_cov):
    
    # a data generation function, not used here
    
    xcs = []
    xts = []
    
    for i in np.linspace(0.1, 0.4, num_cov):
        xcs.append(np.random.binomial(1, i, size=num_control))   # data for conum_treatedrol group
        xts.append(np.random.binomial(1, 1.-i, size=num_treated))   # data for treatmenum_treated group
        
    xc = np.vstack(xcs).T
    xt = np.vstack(xts).T
    
    errors1 = np.random.normal(0, 1, size=num_control)    # some noise
    errors2 = np.random.normal(0, 1, size=num_treated)    # some noise
    
    #dense_bs_sign = np.random.choice([-1,1], num_cov_dense)
    dense_bs = [ (1./2)**i for i in range(num_cov) ]

    yc = np.dot(xc, np.array(dense_bs)) #+ errors1     # y for conum_treatedrol group 
    
    yt = np.dot(xt, np.array(dense_bs)) + 10 #+ errors2    # y for treated group 
        
    df1 = pd.DataFrame(np.hstack([xc]), 
                       columns = range(num_cov))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt]), 
                       columns = range(num_cov ) ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs


# In[50]:


# In[25]:


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


# In[51]:


# In[26]:

def data_generation_dense_2(num_control, num_treated, num_cov_dense, num_covs_unimportant, 
                            control_m = 0.1, treated_m = 0.9):
    
    # the data generating function that we will use. include second order information
    
    xc = np.random.binomial(1, 0.5, size=(num_control, num_cov_dense))   # data for conum_treatedrol group
    xt = np.random.binomial(1, 0.5, size=(num_treated, num_cov_dense))   # data for treatmenum_treated group
        
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

    xc2 = np.random.binomial(1, control_m, size=(num_control, num_covs_unimportant))   # unimportant covariates for control group
    xt2 = np.random.binomial(1, treated_m, size=(num_treated, num_covs_unimportant))   # unimportant covariates for treated group
        
    df1 = pd.DataFrame(np.hstack([xc, xc2]), 
                       columns = range(num_cov_dense + num_covs_unimportant))
    df1['outcome'] = yc
    df1['treated'] = 0

    df2 = pd.DataFrame(np.hstack([xt, xt2]), 
                       columns = range(num_cov_dense + num_covs_unimportant ) ) 
    df2['outcome'] = yt
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df, dense_bs, treatment_eff_coef




# In[27]:

# In[52]:

def data_generation(num_control, num_treated, num_cov, control_m = 0.3, treated_m = 0.7):
    
    # a data generation function. not used
    
    x1 = np.random.binomial(1, control_m, size=(num_control, num_cov) )   # data for conum_treatedrol group
    x2 = np.random.binomial(1, treated_m, size=(num_treated, num_cov) )   # data for treatmenum_treated group

    errors1 = np.random.normal(0, 0.005, size=num_control)    # some noise
    errors2 = np.random.normal(0, 0.005, size=num_treated)    # some noise

    mus = [] 
    sigmas = [] 
    bs = [] 
    for i in range(num_cov):     
        mus.append(i)
        sigmas.append(1./(i**2+1))            # generating weights of covariates for the outcomes

    bs = [np.random.normal(mus[i], sigmas[i]) for i in range(len(sigmas))]  # bs are the weights for the covariates for generating ys

    y1 = np.dot(x1, np.array(bs)) + errors1     # y for control group 
    y2 = np.dot(x2, np.array(bs)) + 1 + errors2    # y for treated group 

    df1 = pd.DataFrame(x1, columns=[i for i in range(num_cov)])
    df1['outcome'] = y1
    df1['treated'] = 0

    df2 = pd.DataFrame(x2, columns=[i for i in range(num_cov)] ) 
    df2['outcome'] = y2
    df2['treated'] = 1

    df = pd.concat([df1,df2])
    df['matched'] = 0
  
    return df







# In[28]:

# In[53]:

def match(df, covs, covs_max_list, treatment_indicator_col = 'treated', match_indicator_col = 'matched'):
    
    
    arr_slice_wo_t = df[covs].values # the covariates values as a matrix
    arr_slice_w_t = df[ covs + [treatment_indicator_col] ].values # the covariate values together with the treatment indicator as a matrix
        
    lidx_wo_t = np.dot( arr_slice_wo_t, np.array([ covs_max_list[i]**(len(covs_max_list) - 1 - i) for i in range(len(covs_max_list))]) ) # matrix multiplication, get a unique number for each unit
    lidx_w_t = np.dot( arr_slice_w_t, np.array([ covs_max_list[i]**(len(covs_max_list) - i) for i in range(len(covs_max_list))] +                                               [1]
                                              ) ) # matrix multiplication, get a unique number for each unit with treatment indicator
        
    _, unqtags_wo_t, counts_wo_t = np.unique(lidx_wo_t, return_inverse=True, return_counts=True) # count how many times each number appears
    _, unqtags_w_t, counts_w_t = np.unique(lidx_w_t, return_inverse=True, return_counts=True) # count how many times each number appears (with treatment indicator)
    
    match_indicator = ~(counts_w_t[unqtags_w_t] == counts_wo_t[unqtags_wo_t]) # a unit is matched if and only if the counts don't agree
        
    return match_indicator, lidx_wo_t[match_indicator]

# In[54]:


# In[29]:

# function to compute the balancing factor
def balancing_factor(df, match_indicator, tradeoff = 0.1):
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


# In[ ]:


# In[30]:


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
    


# In[248]:


# In[31]:


# match_quality, the larger the better
def match_quality(BF, PE):
    ''' 
    '''    
    res = BF + PE
    
    return  res
    
# In[55]:



# In[32]:

def get_CATE_bit(df, match_indicator, index):
    d = df[ match_indicator ]
    if index is None: # when index == None, nothing is matched
        return None
    d.loc[:,'grp_id'] = index
    res = d.groupby(['grp_id', 'treated'])['outcome'].aggregate([np.size, np.mean]) # we do a groupby to get the statistics
    return res


# In[33]:


# In[56]:

def recover_covs(d, covs, covs_max_list, binary = True):
        
    ind = d.index.get_level_values(0)
    ind = [ num2vec(ind[i], covs_max_list) for i in range(len(ind)) if i%2==0]

    df = pd.DataFrame(ind, columns=covs ).astype(int)
    mean_list = list(d['mean'])
    size_list = list(d['size'])
        
    effect_list = [mean_list[2*i+1] - mean_list[2*i] for i in range(len(mean_list)/2) ]
    df.loc[:,'effect'] = effect_list
    df.loc[:,'size'] = [size_list[2*i+1] + size_list[2*i] for i in range(len(size_list)/2) ]
    
    return df



# In[34]:


def cleanup_result(res_all):
    res = []
    for i in range(len(res_all)):
        r = res_all[i]
        if not r[1] is None:
            res.append(recover_covs( r[1], r[0][0], r[0][1] ) )
    return res


# In[35]:

def num2vec(num, covs_max_list):
    res = []
    for i in range(len(covs_max_list)):
        num_i = num/covs_max_list[i]**(len(covs_max_list)-1-i)
        res.append(num_i)
        
        if (num_i == 0) & (num%covs_max_list[i]**(len(covs_max_list)-1-i) == 0):
            res = res + [0]*(len(covs_max_list)-1-i)
            break
        num = num - num_i* covs_max_list[i]**(len(covs_max_list)-1-i)
    return res


# In[57]:



# In[36]:

def run_bit(df, holdout, covs, covs_max_list,threshold, tradeoff_param = 0.1):
    constant_list = ['outcome', 'treated']
    
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


    matching_res = [[( cur_covs, cur_covs_max_list, None, match_indicator, index), res]] # result on first level, None says nothing is dropped

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
        print(cur_covs)
        
        #best_score = np.inf
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
        matching_res.append([best_res, new_matching_res])

        
        s = time.time()
        df = df[~ best_res[-2] ]
        timings[2] = timings[2] + time.time() - s     
        
        units_left = len(df)
#        print units_left
        if units_left <= threshold: 
            print('reached threshold')  
            break
            
    return (timings, cleanup_result(matching_res), nb_steps, level_scores, drops,nb_match_units )




# In[37]:

#example
#d = data_generation_dense_2(10000, 10000, 1,5, control_m = 0.1, treated_m = 0.9)
#df = d[0] 
#holdout,_,_ = data_generation_dense_2(10000, 10000, 1,5, control_m = 0.1, treated_m = 0.9)


# In[38]:

#res = run_bit(df, holdout, range(6), [2]*6, threshold = 0, tradeoff_param = 0.001)


# In[ ]:



