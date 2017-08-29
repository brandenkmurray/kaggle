
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 600)
from hyperopt import fmin, tpe, hp, partial
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score as auc
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import Parallel, delayed
from scipy.stats import rankdata, skew, kurtosis, entropy
from itertools import combinations
import sys
#XGB path
sys.path.append('/Users/johnsmith/Documents/Git/xgboost/python-package')
import xgboost as xgb
import numba
from sklearn.neighbors import NearestNeighbors
#from RGF import *
from os import listdir
from os.path import isfile, join
import shutil
import time

import gc

def gini(y_true):
    l = y_true.shape[0]
    y_sorted_true = np.sort(y_true)
    random = np.arange(1,l+1)/l
    Lorentz_true = np.cumsum(y_sorted_true) / np.sum(y_sorted_true)
    return 2.0*np.sum(random - Lorentz_true)/l

def normalized_gini(y_true,y_pred):
    l = y_pred.shape[0]
    y_sorted_scor = y_true[y_pred.argsort()]
    y_sorted_true = np.sort(y_true)
    random = np.arange(1,l+1)/l
    Lorentz_scor = np.cumsum(y_sorted_scor) / np.sum(y_sorted_scor)
    Lorentz_true = np.cumsum(y_sorted_true) / np.sum(y_sorted_true)
    return np.sum(Lorentz_scor - random) / np.sum(Lorentz_true - random)

def kappa_grad_hess(y,p):
    norm = p.dot(p) + y.dot(y)
    grad = -2*y/norm + 4*p*np.dot(y,p)/(norm**2)
    hess = 8*p*y/(norm**2) + 4*np.dot(y,p)/(norm**2)  - (16*p**2*np.dot(y,p))/(norm**3)
    return grad, hess

def kappa_relaxed(preds, dtrain):
    target = dtrain.get_label()
    #target -=y_mean
    #preds -=y_mean
    return 'kappa' ,  -2 * target.dot(preds) / (target.dot(target) + preds.dot(preds))

def myauc(preds, dtrain):
    target = dtrain.get_label()

    return 'auc' ,  -auc(target,preds)

def my_kappa(y,p):
    return -2 * y.dot(p) / (y.dot(y) + p.dot(p))


def ginierror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', 1-normalized_gini(labels,preds)

def qwk(preds, dtrain):
    target = dtrain.get_label()+y_mean
    
    return 'qwk' ,  -quadratic_weighted_kappa(np.round(target),np.clip(np.round(preds+y_mean),1,8))

def make_avg(train,test,all_data,rs=3):
    y = train.target.values
    y_mean = np.mean(y)
    cols = list(train.columns)
    cols.remove('ID')
    cols.remove('target')
    
    cols_unique_thr = []
    cols_unique_thr_mt = []
    cols_unique_thr_cnt = []
    short_cols = []
    for col in cols:
        l_all = all_data[col].unique().shape[0]
        cols_unique_thr.append(col)
        cols_unique_thr_mt.append(col+'_mean_target')
        cols_unique_thr_cnt.append(col+'_cnt')
    
    train_new = pd.concat([train,pd.DataFrame(columns=cols_unique_thr_mt+cols_unique_thr_cnt)],axis=1)  
    skf = StratifiedKFold(y,5, shuffle=True,random_state=rs)
    iter_num = 0
    for tr_ind, val_ind in skf:
        X_tr, X_val = train.iloc[tr_ind], train.iloc[val_ind]
        new_columns = []
        for col in cols_unique_thr:
            counts = X_val[col].map(all_data.groupby(col).ID.size())
            means = X_val[col].map(X_tr.groupby(col).target.mean())
            X_val[col+'_mean_target'] = means#y_mean+(2.0/np.pi)*np.arctan(np.log(counts)) * (means-y_mean) #
            X_val[col+'_cnt'] = counts
        train_new.iloc[val_ind] = X_val
        print(iter_num)
        iter_num+=1
    
    train = train_new
    train.fillna(y_mean,inplace=True)
    
    
    test = pd.concat([test,pd.DataFrame(1.,index = test.index,columns=cols_unique_thr_mt+cols_unique_thr_cnt)],axis=1)

    for col in cols_unique_thr:
        counts = test[col].map(all_data.groupby(col).ID.size())
        means = test[col].map(train.groupby(col).target.mean())
        test[col+'_mean_target'] =means#y_mean+(2.0/np.pi)*np.arctan(np.log(counts)) * (means-y_mean) #
        test[col+'_cnt'] = counts

    train.fillna(y_mean,inplace=True)
    test.fillna(y_mean,inplace=True)
    test = test[list(train.columns)]

    train_thr = np.array(train[cols_unique_thr_mt+cols_unique_thr_cnt])
    test_thr = np.array(test[cols_unique_thr_mt+cols_unique_thr_cnt])
    return train_thr, test_thr


# In[ ]:

def find_delimiter(df, col):
    """
    Function that trying to find an approximate delimiter used for scaling.
    So we can undo the feature scaling.
    """
    vals = df[col].dropna().sort_values().round(8)
    vals = pd.rolling_apply(vals, 2, lambda x: x[1] - x[0])
    vals = vals[vals > 0.000001]
    return vals.value_counts().idxmax() 

to_drop_cols = ['v8','v17','v23','v25','v26','v36','v37','v46','v53','v54','v63','v73','v75','v79','v81','v82','v89',
                'v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128',
               'v9','v12','v40','v67','v68','v83','v85','v86','v88','v94','v96','v106','v111','v115','v121']

to_drop_cols = ['v8','v17','v23','v25','v26','v36','v37','v46','v53','v54','v63','v73','v75','v79','v81','v82','v89',
                'v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']



count_cols = ['v129','v72','v62','v38'] #v72 = v129 + v62 + v38
#v22 18k unique
#v52 может месяц? v91 - день недели? 

train = pd.read_csv('/media/branden/SSHD1/kaggle/bnp/train.csv')
test = pd.read_csv('/media/branden/SSHD1/kaggle/bnp/test.csv')
all_data = pd.concat([train,test], ignore_index=True)
#num_vars = ['v1','v2','v4','v5','v6','v7','v8','v9','v10','v13','v16','v18','v19','v20','v21','v23','v25',
#            'v28','v29','v32','v33','v38','v39','v41','v42','v44','v49','v50','v51','v55','v59','v62','v70',
#            'v76','v78','v80','v81','v82','v84','v85','v87','v88',
#            'v90','v93','v94','v95','v98','v99','v101','v102','v104','v114','v115','v121', 'v124','v129','v131']
#to_drop_cols = [col for col in all_data.select_dtypes(include=['float64','int64']).columns if col not in num_vars]
#to_drop_cols.remove('target')
#to_drop_cols.remove('ID')
all_data.drop(to_drop_cols,axis=1,inplace=True)
num_train = train.shape[0]
num_vars = list(all_data.select_dtypes(include=['float64','int64']).columns)
num_vars.remove('target')
num_vars.remove('ID')
for col in count_cols:
    num_vars.remove(col)
cat_cols = list(all_data.select_dtypes(include=['object']).columns)
#позже разбить NA на 2 группы - из-за джойна и настоящие
all_data[cat_cols] = all_data[cat_cols].apply(lambda x: pd.factorize(x,na_sentinel=-1)[0])
cat_cols.extend(count_cols)
all_data['v22_cnt'] = all_data.groupby('v22').ID.transform('size')

'''
num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v10','v11',
            'v13', 'v14', 'v15', 'v16', 'v18', 'v19','v20',
            'v21', 'v27', 'v28', 'v29','v32','v33', 'v34', 'v35', 'v38',
            'v39','v41','v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50','v51',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65',
            'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v84', 
            'v87', 'v90', 'v93', 'v97', 'v98', 
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v114',
            'v120', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']

num_vars = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11',
            'v12', 'v13', 'v14', 'v15', 'v16', 'v18', 'v19', 'v20',
            'v21', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v38',
            'v39', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v48', 'v49', 'v50','v51',
            'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v62', 'v64', 'v65', 'v67',
            'v68', 'v69', 'v70', 'v72', 'v76', 'v77', 'v78', 'v80', 'v83', 'v84', 
            'v85', 'v86', 'v87', 'v88', 'v90', 'v93', 'v94', 'v96', 'v97', 'v98', 
            'v99', 'v100', 'v101', 'v102', 'v103', 'v104', 'v106', 'v111', 'v114',
            'v115', 'v120', 'v121', 'v122', 'v126', 'v127', 'v129', 'v130', 'v131']
'''


all_data['v10'] = np.round(all_data['v10']/0.0218818357511,0)
cat_cols.append('v10')
num_vars.remove('v10')
#num_vars.remove('v50')
all_data[num_vars] = all_data[num_vars].apply(lambda x: np.round(x,4))
print(all_data.shape[1],len(cat_cols),len(num_vars))


# In[ ]:

#%%capture --no-stdout --no-display
from scipy.stats import mode

def xgb_worker(par,X_tr,X_val,y_tr,y_val,seed=1,verbose=True,return_err=True):
    params = par.copy()
    plst = params.items()
    np.random.seed(seed = 123)
    X_tr, y_tr = skshuffle(X_tr, y_tr,random_state=123)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)#,missing = -1.0)
    dvalid = xgb.DMatrix(X_val, label=y_val)#,missing = -1.0)
    evallist = [(dtrain, 'train'),(dvalid, 'eval')]
    
    num_round = 100000
    model = xgb.train( plst, dtrain,num_round,
                      evals=evallist,early_stopping_rounds=50)
    pred = model.predict(dvalid,ntree_limit=model.best_iteration)
    err = model.best_score   
    
    if verbose:
        f = open('logshyper.txt','a')
        f.write(str(err)+' '+str(model.best_iteration)+' '+str(int(params['max_depth']))+' '+
                str(np.round(params['subsample'],3))
                +' '+str(np.round(params['colsample_bytree'],3))+' '+str(params['eta'])+' '+str(params['gamma'])
                +' '+str(params['min_child_weight'])+'\n') 
        f.close()
        print(err,model.best_iteration,int(params['max_depth']),np.round(params['subsample'],3),
              np.round(params['colsample_bytree'],3),np.round(params['min_child_weight'],1),
              params['eta'],np.round(params['gamma'],1))
    if return_err:
        return err
    else:
        return pred, model

def make_xgb(params,X_tr,y_tr,nr,rs=1):
    par = params.copy()
    par['seed'] = rs
    plst = par.items()
    np.random.seed(seed = rs+123)
    num_round=nr
    X_tr, y_tr = skshuffle(X_tr, y_tr,random_state=rs+123)
    noise = np.random.normal(0,0.5,len(y_tr))
    dtrain = xgb.DMatrix(X_tr, label=y_tr)#+noise)#,missing=-1.0)
    #dtest = xgb.DMatrix(test)#,missing=-1.0)
    model = xgb.train( plst, dtrain,num_round)#,obj=KappaRelaxedObjective())
    #pred = model.predict(dtest)
    return model

import numba

@numba.jit
def get_hist(arr):
    res = np.zeros((len(arr),20))
    for i in range(len(arr)):
        res[i] = np.histogram(arr[i],20,density=False)[0]
    return res
    
def get_stats(arr):
    top10 = np.sort(arr,axis=1)[:,:10]
    low10 = np.sort(arr,axis=1)[:,-10:]
    hist = get_hist(arr)
    m = np.mean(arr,axis=1)
    med = np.median(arr,axis=1)
    p1 = np.percentile(arr,1,axis=1)
    p5 = np.percentile(arr,5,axis=1)
    p10 = np.percentile(arr,10,axis=1)
    p25 = np.percentile(arr,25,axis=1)
    p75 = np.percentile(arr,75,axis=1)
    p90 = np.percentile(arr,90,axis=1)
    p95 = np.percentile(arr,95,axis=1)
    p99 = np.percentile(arr,99,axis=1)
    s = np.std(arr,axis=1)
    sk = skew(arr,axis=1)
    kurt = kurtosis(arr,axis=1)
    #ent = -np.sum(arr * np.log(arr), axis=1)
    
    return np.c_[top10,low10,hist,m,med,p1,p5,p10,p25,p75,p90,p95,p99,s,sk,kurt]
    

xgb_par = {'eta' : 0.01,'max_depth' : 5,'subsample' : 0.8,'colsample_bytree' : 0.75,'objective':'binary:logistic',
           'min_child_weight':2, 'seed': 1,'nthread' : 1,'silent' : 1, 'eval_metric':'logloss','gamma':0.42}#350

xgb_par = {'eta' : 0.01,'max_depth' : 8,'subsample' : 0.9,'colsample_bytree' : 0.5,'objective':'binary:logistic',
           'min_child_weight':2, 'seed': 1,'nthread' : 1,'silent' : 1, 'eval_metric':'logloss','gamma':0.7}#350

dd={}
for col in num_vars:
    dd[col] = ['min','max','mean','std','sum','size']
f_dict_red = pickle.load( open( "f_dict_red.p", "rb" ) )
all_data.fillna(-1,inplace=True)

g_cols = []
v22_int = ['ID','target']
#['min','max','mean','std','sum','size']

good_pairs = ['v113v22','v125v52','v125v91','v22v52', 'v22v56','v22v91','v31v66', 'v47v56','v52v91']
for i,pair in enumerate(combinations(cat_cols,2)):
    c1 = pair[0]
    c2 = pair[1]
    if c1+c2 not in good_pairs:
        continue
    g = all_data[[c1,c2]+['v50']].groupby([c1,c2],as_index=False).agg({'v50':['mean']})
    g.columns = [col[0] if col[-1]=='' else c1+c2+col[0]+'_'+col[-1] for col in g.columns.values]
    all_data = pd.merge(all_data,g,how='left',on=[c1,c2])
    g_cols.extend(list(set(g.columns)-set([c1,c2])))
    if i%100==0:
        print(i)

train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
y = train.target.values
train.drop(['target','ID'],axis=1,inplace=True)
train.fillna(-1,inplace=True)
test_idx = test.ID.values
test.drop(['target','ID'],axis=1,inplace=True)

test.fillna(-1,inplace=True)


#X_train = train.values
skf = StratifiedKFold(y,5,random_state=660)#StratifiedKFold(y,4,shuffle=True,random_state=660)
pred_train = np.zeros(len(y))
num_fold = 0
for tr_ind, val_ind in skf:
    X_tr, X_val = train.iloc[tr_ind].values, train.iloc[val_ind].values
    y_tr, y_val = y[tr_ind], y[val_ind]
    X_trm = np.load('datasets/v22means_tr'+str(num_fold)+'.npy')
    X_valm = np.load('datasets/v22means_val'+str(num_fold)+'.npy')
    X_trm = np.mean(y)+np.multiply(X_trm[:,:630]-np.mean(y),(2.0/np.pi)*np.arctan(np.log(X_trm[:,630:])))
    X_valm = np.mean(y)+np.multiply(X_valm[:,:630]-np.mean(y),(2.0/np.pi)*np.arctan(np.log(X_valm[:,630:])))
    X_tr = np.c_[train.iloc[tr_ind].values,X_trm]#,get_stats(X_trm)]
    X_val = np.c_[train.iloc[val_ind].values,X_valm]#,get_stats(X_valm)]
    #
    p, model=xgb_worker(xgb_par,X_tr,X_val,y_tr,y_val,seed=1,verbose=False,return_err=False)
    break
    ll = Parallel(n_jobs=4)(delayed(make_xgb)(xgb_par,X_tr,y_tr, 850,rs=i+123) 
                        for i in range(4))
    X_val = xgb.DMatrix(np.c_[train.iloc[val_ind].values,np.load('datasets/v22means_val'+str(num_fold)+'.npy')])
    p = np.zeros(len(y_val))
    for j in range(len(ll)):
        p+=ll[j].predict(X_val)
    p/=len(ll)
    del X_val       
    pred_train[val_ind] = p
    num_fold+=1
#print(log_loss(y,pred_train))
#np.save('metafeatures/train/xgb_02.npy',pred_train)
'''
train = np.c_[train.values,np.load('datasets/v22means_train.npy'),np.load('datasets/v22means2_train.npy')]
print(train.shape)
ll = Parallel(n_jobs=3)(delayed(make_xgb)(xgb_par,train,y, 850,rs=i+123) 
                    for i in range(6))
del train
test = xgb.DMatrix(np.c_[test.values,np.load('datasets/v22means_test.npy'),np.load('datasets/v22means2_test.npy')])
pred_test = np.zeros(len(test_idx))
for j in range(len(ll)):
    pred_test+=ll[j].predict(test)
pred_test/=len(ll)

df = pd.DataFrame(np.c_[test_idx,pred_test],columns=['ID','PredictedProb'])
df['ID'] = df['ID'].astype(np.int32)
df.to_csv('xgb_sub_03.csv',index=False)
'''


# In[ ]:



