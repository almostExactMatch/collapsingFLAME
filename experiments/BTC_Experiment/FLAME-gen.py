import numpy as np
import pandas as pd
import pyodbc
import time
import pickle
import operator
from operator import itemgetter
from joblib import Parallel, delayed

from sklearn import linear_model

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sqlalchemy import create_engine

import matplotlib.pyplot as plt

import psycopg2
from sklearn.utils import shuffle

import sql
from sklearn.utils import shuffle
from sklearn import svm
from decimal import *
import random

# this function takes the current covariate list, the covariate we consider dropping, name of the data table, 
# name of the holdout table, the threshold (below which we consider as no match), and balancing regularization
# as input; and outputs the matching quality
def score_tentative_drop_c(cov_l, c, db_name, holdout_df, thres = 0, tradeoff = 0.000):
    covs_to_match_on = set(cov_l) - {c} # the covariates to match on
    
    # the flowing query fetches the matched results (the variates, the outcome, the treatment indicator)
    s = time.time()
   
    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("treated")>'0' and sum("treated")<count(*) 
        )
        (SELECT {1}, treated, outcome
        FROM {3}
        WHERE EXISTS 
        (SELECT 1
        FROM temp 
        WHERE {2}
        )
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_to_match_on ]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_to_match_on ]),
                   db_name
                  ) )
    res = np.array(cur.fetchall())
    
    time_match = time.time() - s
    
    s = time.time()
    # the number of unmatched treated units
    cur.execute('''select count(*) from {} where "matched" = 0 and "treated" = 0'''.format(db_name))
    num_control = cur.fetchall()
    # the number of unmatched control units
    cur.execute('''select count(*) from {} where "matched" = 0 and "treated" = 1'''.format(db_name))
    num_treated = cur.fetchall()
    time_BF = time.time() - s
    
    # fetch from database the holdout set
    
    s = time.time() # the time for fetching data into memory is not counted if use this
    
    # below is the regression part for PE
    ridge_c = Ridge(alpha=1)
    ridge_t = Ridge(alpha=1)
    
    holdout = holdout_df.copy()
    holdout = holdout[ ["{}".format(c) for c in covs_to_match_on] + ['treated', 'outcome']]
   
    mse_t = np.mean(cross_val_score(ridge_t, holdout[holdout['treated'] == 1].iloc[:,:-2], 
                                holdout[holdout['treated'] == 1]['outcome'] , scoring = 'neg_mean_squared_error' ) )
        
    mse_c = np.mean(cross_val_score(ridge_c, holdout[holdout['treated'] == 0].iloc[:,:-2], 
                                holdout[holdout['treated'] == 0]['outcome'], scoring = 'neg_mean_squared_error' ) )
    
    PE = mse_t + mse_c
    
    time_PE = time.time() - s
    
    num_t = holdout[holdout['treated'] == 1]['outcome'].shape[0]
    num_c = holdout[holdout['treated'] == 0]['outcome'].shape[0]
    
    if len(res) == 0:
        return (PE, time_match, time_PE, time_BF)
    else:        
        return ( tradeoff * (float(len(res[res[:,-2]==0]))/num_control[0][0] + float(len(res[res[:,-2]==1]))/num_treated[0][0]) + PE, 
                time_match, time_PE, time_BF)
    
def update_matched(covs_matched_on, db_name, level):
    cur.execute('''with temp AS 
        (SELECT 
        {0}
        FROM {3}
        where "matched"=0
        group by {0}
        Having sum("treated")>'0' and sum("treated")<count(*) 
        )
        update {3} set "matched"={4}
        WHERE EXISTS
        (SELECT {0}
        FROM temp
        WHERE {2} and {3}."matched" = 0
        )
        '''.format(','.join(['"{0}"'.format(v) for v in covs_matched_on]),
                   ','.join(['{1}."{0}"'.format(v, db_name) for v in covs_matched_on]),
                   ' AND '.join([ '{1}."{0}"=temp."{0}"'.format(v, db_name) for v in covs_matched_on ]),
                   db_name,
                   level
                  ) )
    conn.commit()
    return

def get_CATE_db(cov_l, db_name, level):
    cur.execute(''' select {0}, array_agg(index0),avg(outcome * 1.0), count(*)
                    from {1}
                    where matched = {2} and treated = 0
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_c = cur.fetchall()
       
    cur.execute(''' select {0}, array_agg(index0),avg(outcome * 1.0), count(*)
                    from {1}
                    where matched = {2} and treated = 1
                    group by {0}
                    '''.format(','.join(['"{0}"'.format(v) for v in cov_l]), 
                              db_name, level) )
    res_t = cur.fetchall()
     
    if (len(res_c) == 0) | (len(res_t) == 0):
        return None
    
    cov_l = list(cov_l)

    result = pd.merge(pd.DataFrame(np.array(res_c), columns=['{}'.format(i) for i in cov_l]+['index_c','effect_c', 'count_c']), 
                  pd.DataFrame(np.array(res_t), columns=['{}'.format(i) for i in cov_l]+['index_t','effect_t', 'count_t']), 
                  on = ['{}'.format(i) for i in cov_l], how = 'inner') 
    
    result_df = result[['{}'.format(i) for i in cov_l] + ['index_c', 'index_t', 'effect_c', 'effect_t', 'count_c', 'count_t']]
    
    result_df['index'] = result_df['index_c'] + result_df['index_t']
    result_df['mean'] = result_df['effect_t'] - result_df['effect_c']

    result_df = result_df[['index','mean']]
    
    if result_df is None or result_df.empty:
        return None
    
    return result_df

def process_data():
    #parse data
    df = pd.read_csv('~/Desktop/MyBTCData_R2.csv', index_col=0, parse_dates=True)
    df = df.rename(columns={'BTC': 'treated', 'outcome_matrix$ANY_NDRU': 'outcome'})
    df_treated = df.loc[:,'treated']
    df = df.drop('treated',1)
    df_outcome = df.loc[:,'outcome']
    df = df.drop('outcome',1)
    shape = df.shape 
    row_num = shape[0]
    col_num = shape[1]
    df.columns = np.arange(col_num)
    df.columns = df.columns.astype(str) 

    #merge covariates and outcomes
    df = pd.concat([df, df_treated, df_outcome], axis=1)
    print(df.corr())   
    for label in df:
        if label == 'outcome':
            df[label][df[label] == 0] = -1
    df['outcome'] = df['outcome'].astype('object')        
    df['matched'] = 0
    
    df = df.reset_index()
    df['index0'] = df.index
    df = df.drop('index',1)
    df.to_csv("data.csv")
    
    return df,df

# In[7]:

def run_db(db_name, holdout_df, num_covs, reg_param = 0.1):
    cur.execute('update {0} set matched = 0'.format(db_name)) # reset the matched indicator to 0
    conn.commit()

    covs_dropped = [] # covariate dropped
    ds = []
    score_list = []
    
    level = 1
    
    timings = [0]*5 # first entry - match (groupby and join), 
                    # second entry - regression (compute PE), 
                    # third entry - compute BF, 
                    # fourth entry - keep track of CATE, 
                    # fifth entry - update database table (mark matched units). 
    
    cur_covs = range(num_covs) # initialize the current covariates to be all covariates

    # make predictions and save to disk
    s = time.time()
    update_matched(cur_covs, db_name, level) # match without dropping anything
    timings[4] = timings[4] + time.time() - s
        
    s = time.time()
    d = get_CATE_db(cur_covs, db_name, level) # get CATE without dropping anything
    timings[3] = timings[3] + time.time() - s
    score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, None, db_name, 
                                                              holdout_df, tradeoff = 0.000)    
    ds.append(d)
    
    timings[4] = timings[4] + time.time() - s

    while len(cur_covs)>1:
        level += 1
        
        # the early stopping conditions
        cur.execute('''select count(*) from {} where "matched"=0 and "treated"=0'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            print("early stop")
            break

        cur.execute('''select count(*) from {} where "matched"=0 and "treated"=1'''.format(db_name))
        if cur.fetchall()[0][0] == 0:
            print("early stop")
            break
        
        best_score = -np.inf
        cov_to_drop = None
        
        cur_covs = list(cur_covs)
        for c in cur_covs:
            score,time_match,time_PE,time_BF = score_tentative_drop_c(cur_covs, c, db_name, 
                                                                      holdout_df, tradeoff = 0.000)
            
            timings[0] = timings[0] + time_match
            timings[1] = timings[1] + time_PE
            timings[2] = timings[2] + time_BF
            if score > best_score:
                best_score = score
                cov_to_drop = c

        cur_covs = set(cur_covs) - {cov_to_drop} # remove the dropped covariate from the current covariate set
        
        s = time.time()
        update_matched(cur_covs, db_name, level)
        timings[4] = timings[4] + time.time() - s
        score_list.append(best_score)
        
        s = time.time()
        d = get_CATE_db(cur_covs, db_name, level)
        timings[3] = timings[3] + time.time() - s
        
        ds.append(d)
        
        covs_dropped.append(cov_to_drop) # append the removed covariate at the end of the covariate 
    
    return ds, score_list

if __name__ == '__main__':
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='yaoyj11 '")
    cur = conn.cursor()  
    
    engine = create_engine('postgresql+psycopg2://postgres:yaoyj11 @localhost/postgres')
    table_name = 'flamedb'
    cur.execute('drop table if exists {}'.format(table_name))
    conn.commit()
    
    train,test = process_data()
    train.to_sql(table_name, engine)
    
    res = run_db(table_name, test, train.shape[1]-4)   
    pickle.dump(res, open('FLAME-gen-result', 'wb'))
