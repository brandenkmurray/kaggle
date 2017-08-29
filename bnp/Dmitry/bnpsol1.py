
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 600)
from hyperopt import fmin, tpe, hp, partial
from fastFM import als, sgd, mcmc
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score as auc
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.stats import rankdata
from itertools import combinations
from scipy import sparse
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

def make_avg(train,test,all_data,rs=3,loo=False):
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
    if not loo:
        train_new = pd.concat([train,pd.DataFrame(columns=cols_unique_thr_mt+cols_unique_thr_cnt)],axis=1)  
        skf = StratifiedKFold(y,4, shuffle=True,random_state=rs)
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
    else:
        train = pd.concat([train,pd.DataFrame(1.,index = train.index,columns=cols_unique_thr_mt+cols_unique_thr_cnt)],axis=1)

        for col in cols_unique_thr:
            counts = train[col].map(all_data.groupby(col).ID.size())
            cnt = train[col].map(train.groupby(col).ID.size())
            means = train[col].map(train.groupby(col).target.mean())
            train[col+'_mean_target'] = (means*cnt-train['target'])/(cnt-1)
            train[col+'_cnt'] = counts
    
    train.fillna(y_mean,inplace=True)
    
    
    test = pd.concat([test,pd.DataFrame(1.,index = test.index,columns=cols_unique_thr_mt+cols_unique_thr_cnt)],axis=1)

    for col in cols_unique_thr:
        counts = test[col].map(all_data.groupby(col).ID.size())
        means = test[col].map(train.groupby(col).target.mean())
        test[col+'_mean_target'] =means#y_mean+(2.0/np.pi)*np.arctan(np.log(counts)) * (means-y_mean) #
        test[col+'_cnt'] = counts

    test.fillna(y_mean,inplace=True)
    test = test[list(train.columns)]

    train_thr = np.array(train[cols_unique_thr_mt+cols_unique_thr_cnt])
    test_thr = np.array(test[cols_unique_thr_mt+cols_unique_thr_cnt])
    return train_thr, test_thr


# In[ ]:

to_drop_cols = ['v107']
count_cols = ['v129','v72','v62','v38'] #v72 = v129 + v62 + v38
#v22 18k unique
#v52 может месяц? v91 - день недели? 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[ ]:

train.head(10).transpose()


# In[ ]:

for col in train.columns:
    print(col,len(train[col].value_counts()))


# In[ ]:

col = 'v10'
plt.hist(train[~train[col].isnull()][col]/0.0218818357511,200);


# In[ ]:

cnt = train['v22'].map(train.groupby('v22').v50.min())
plt.scatter(cnt,train.target+np.random.normal(0,0.01,len(train)))


# In[2]:

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

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
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

LR = LogisticRegression(C=0.001)

LR.fit(X_tr,y_tr)
p=LR.predict_proba(X_val)
print(log_loss(y_val,p))


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

def make_xgb(params,X_tr,y_tr,test,nr,rs=1):
    par = params.copy()
    par['seed'] = rs
    plst = par.items()
    np.random.seed(seed = rs+123)
    num_round=nr
    X_tr, y_tr = skshuffle(X_tr, y_tr,random_state=rs+123)
    noise = np.random.normal(0,0.5,len(y_tr))
    dtrain = xgb.DMatrix(X_tr, label=y_tr)#+noise)#,missing=-1.0)
    dtest = xgb.DMatrix(test)#,missing=-1.0)
    model = xgb.train( plst, dtrain,num_round)#,obj=KappaRelaxedObjective())
    pred = model.predict(dtest)
    return pred
    
def optimize(space,train,test,y,y_val):
    
    best = fmin(partial(xgb_worker,X_tr=train,X_val=test,y_tr=y,y_val=y_val), space, algo=tpe.suggest, max_evals=50)

    print (best)
    return best



space = {
         'eta' : 0.01,#hp.quniform('eta', 0.01, 0.1, 0.005),
         'max_depth' : hp.quniform('max_depth', 5, 10, 1),
         'subsample' : hp.quniform('subsample', 0.6, 0.9, 0.05),
         'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 0.8, 0.05),
         'min_child_weight' : hp.quniform('min_child_weight', 1, 5, 1),
         'gamma' : hp.quniform('gamma', 0.0, 1, 0.1),
         'objective': 'binary:logistic',
         'seed': 1,
         'eval_metric':'logloss',
         'nthread' : 1,
         'silent' : 1
         }
#отдельные модели для запросов с частотой 1-2

xgb_par = {'eta' : 0.03,'max_depth' : 10,'subsample' : 0.8,'colsample_bytree' : 0.75,'objective':'binary:logistic',
           'min_child_weight':2, 'seed': 1,'nthread' : 1,'silent' : 1, 'eval_metric':'logloss','gamma':0.42}#350

xgb_par = {'eta' : 0.01,'max_depth' : 8,'subsample' : 0.9,'colsample_bytree' : 0.5,'objective':'binary:logistic',
           'min_child_weight':2, 'seed': 1,'nthread' : 1,'silent' : 1, 'eval_metric':'logloss','gamma':0.7}#850




#train.drop(['target','ID'],axis=1,inplace=True)
#all_data.fillna(-1,inplace=True)

dd={}
for col in num_vars:
    dd[col] = ['min','max','mean','std','sum','size']
f_dict_red = pickle.load( open( "f_dict_red.p", "rb" ) )
all_data.fillna(-1,inplace=True)
for c in num_vars:
    all_data[c+'raw'] = all_data[c].copy(deep=True)
    bins = np.array(sorted(list(f_dict_red[c])))
    all_data[c] = np.digitize(all_data[c].values,bins)

g_cols = []
v22_int = ['ID','target','v22']
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

with open('goodcombs2.txt','r') as f:
    gc = f.readlines()
gc = [l.rstrip() for l in gc]
gc = good_pairs+gc

with open('goodcombs3.txt','r') as f:
    gc3 = f.readlines()
gc3 = [l.rstrip() for l in gc3]

all_data.fillna(-1,inplace=True)
for i,pair in enumerate(combinations(cat_cols+num_vars,2)):
    c1 = pair[0]
    c2 = pair[1]
    if 'v22' in [c1,c2]:
        v22_int.append(c1+c2)
        all_data[c1+c2] = all_data[c1].astype(str) + '_' + all_data[c2].astype(str)
    if 'v22' not in [c1,c2] and c1+c2 in gc or c2+c1 in gc:
        v22_int.append(c1+c2+'v22')
        all_data[c1+c2+'v22'] = all_data[c1].astype(str)+'_'+all_data[c2].astype(str)+'_'+all_data['v22'].astype(str)
print(all_data.shape)
for i,triplet in enumerate(combinations(cat_cols+num_vars,3)):
    c1 = triplet[0]
    c2 = triplet[1]
    c3 = triplet[2]
    fc = [c1,c2,c3]
    fc.sort()
    s=''
    for item in fc:
        s+=item
    if 'v22' not in [c1,c2,c3] and s in gc3:
        v22_int.append(c1+c2+c3+'v22')
        all_data[c1+c2+c3+'v22'] = (all_data[c1].astype(str)+'_'+all_data[c2].astype(str)+'_'
                                    +all_data[c3].astype(str)+'_'+all_data['v22'].astype(str))

#all_data = pd.concat([all_data,
#                      pd.DataFrame('s',index = all_data.index,columns=list(set(v22_int)-set(['ID','target'])))],axis=1)
        
print(all_data.shape)     
all_data.drop(num_vars,axis=1,inplace=True)
print(all_data.shape)

    
#train = pd.concat(train_list,axis=1)
#del train_list


train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
y = train.target.values

train.fillna(-1,inplace=True)

test.fillna(-1,inplace=True)


#X_train = train.values
skf = StratifiedKFold(y,5,random_state=660)#StratifiedKFold(y,4,shuffle=True,random_state=660)
pred_train = np.zeros(len(y))

baseline = 0.460423 
num_fold = 0
#4562 - extratrees, 4533 with best cat_cols pairs with v50 stats, 45299 plus na cnt, 4514 after calibration
#[407]	train-logloss:0.291700	eval-logloss:0.459183 cat_cols with v50 stats
#[448]	train-logloss:0.225086	eval-logloss:0.458259 cat_cols pairs with v50 stats
#[347]	train-logloss:0.285436	eval-logloss:0.450468 with target means of v22-cat 2interactions
#[273]	train-logloss:0.284927	eval-logloss:0.447143 with target means of v22-all 2interactions
#[232]	train-logloss:0.286981	eval-logloss:0.440752 with target means of v22-top 2-3interactions
#[185]	train-logloss:0.320399	eval-logloss:0.438401 with target means of v22-top 2-3-4interactions
#0.4573 extratrees no v22, 0.4549 with goodpairs mean, 0.4533 with cc
for tr_ind, val_ind in skf:
    #X_tr, X_val = train.iloc[tr_ind].values, train.iloc[val_ind].values
    y_tr, y_val = y[tr_ind], y[val_ind]
    
    X_tr, X_val = train.copy(deep=True).iloc[tr_ind], train.copy(deep=True).iloc[val_ind]
    tr_thr, val_thr = make_avg(X_tr[v22_int],X_val[v22_int],all_data[v22_int],rs=555,loo=False)
    X_tr,X_val=np.c_[X_tr.drop(v22_int,axis=1).values,tr_thr],np.c_[X_val.drop(v22_int,axis=1).values,val_thr]
    print(X_tr.shape,X_val.shape)
    p, model=xgb_worker(xgb_par,X_tr,X_val,y_tr,y_val,seed=1,verbose=False,return_err=False)
    break
    #np.save('datasets/v22means_tr'+str(num_fold)+'.npy',tr_thr)
    #np.save('datasets/v22means_val'+str(num_fold)+'.npy',val_thr)
    #num_fold+=1
    #continue
    #X_tr = np.c_[X_tr.drop(v22_int,axis=1).values,tr_thr]
    #X_val = np.c_[X_val.drop(v22_int,axis=1).values,val_thr]
    #clf = ExtraTreesClassifier(n_estimators=1000,max_features= 0.35,criterion= 'entropy',min_samples_split= 4,
    #                           min_samples_leaf = 2, n_jobs = 4)
    
'''
np.save('metafeatures/train/etv22_01.npy',pred_train)
pred_v22 = -1*np.ones(len(test))
for v in np.unique(v22_test):
    mask_tr = v22==v
    mask_val = v22_test==v
    if np.sum(mask_tr)<5:
        continue
    if len(np.unique(y[mask_tr]))==1:
        pred_v22[mask_val] = np.unique(y[mask_tr])[0]
        continue
    clf = ExtraTreesClassifier(n_estimators=100,max_features= 0.35,criterion= 'entropy',min_samples_split= 2,
                               min_samples_leaf = 1, n_jobs = 4)
    #cc = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv=10)
    clf.fit(train.values[mask_tr],y[mask_tr])
    p = clf.predict_proba(test.values[mask_val])[:,1]
    pred_v22[mask_val] = p
np.save('metafeatures/test/etv22_01.npy',pred_v22)
'''


# In[ ]:

p_xgb = np.load('metafeatures/train/xgb_02.npy')
print(log_loss(y_val,np.exp(0.1*np.log(p)+0.9*np.log(p_xgb[val_ind]))),log_loss(y_val,p_xgb[val_ind]))


# In[ ]:

def make_avgn(train,test,all_data,rs=3):
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
    
    train_new = pd.concat([train,pd.DataFrame(columns=cols_unique_thr_mt)],axis=1)  
    skf = StratifiedKFold(y,5, shuffle=True,random_state=rs)
    iter_num = 0
    for tr_ind, val_ind in skf:
        X_tr, X_val = train.iloc[tr_ind], train.iloc[val_ind]
        new_columns = []
        for col in cols_unique_thr:
            #counts = X_val[col].map(all_data.groupby(col).ID.size())
            means = X_val[col].map(X_tr.groupby(col).target.mean())
            X_val[col+'_mean_target'] = means#y_mean+(2.0/np.pi)*np.arctan(np.log(counts)) * (means-y_mean) #
            #X_val[col+'_cnt'] = counts
        train_new.iloc[val_ind] = X_val
        print(iter_num)
        iter_num+=1
    
    train = train_new
    train.fillna(y_mean,inplace=True)
    
    
    test = pd.concat([test,pd.DataFrame(1.,index = test.index,columns=cols_unique_thr_mt)],axis=1)

    for col in cols_unique_thr:
        #counts = test[col].map(all_data.groupby(col).ID.size())
        means = test[col].map(train.groupby(col).target.mean())
        test[col+'_mean_target'] =means#y_mean+(2.0/np.pi)*np.arctan(np.log(counts)) * (means-y_mean) #
        #test[col+'_cnt'] = counts

    train.fillna(y_mean,inplace=True)
    test.fillna(y_mean,inplace=True)
    test = test[list(train.columns)]

    train_thr = np.array(train[cols_unique_thr_mt])
    test_thr = np.array(test[cols_unique_thr_mt])
    return train_thr, test_thr

dd={}
for col in num_vars:
    dd[col] = ['min','max','mean','std','sum','size']
f_dict_red = pickle.load( open( "f_dict_red.p", "rb" ) )
all_data.fillna(-1,inplace=True)
for c in num_vars:
    all_data[c+'raw'] = all_data[c].copy(deep=True)
    bins = np.array(sorted(list(f_dict_red[c])))
    all_data[c] = np.digitize(all_data[c].values,bins)

g_cols = []
v22_int = ['ID','target']
v22_int = v22_int+num_vars+cat_cols
#['min','max','mean','std','sum','size']

good_pairs = ['v113v22','v125v52','v125v91','v22v52', 'v22v56','v22v91','v31v66', 'v47v56','v52v91']

with open('goodcombs2.txt','r') as f:
    gc = f.readlines()
gc = [l.rstrip() for l in gc]
gc = good_pairs+gc

with open('goodcombs3.txt','r') as f:
    gc3 = f.readlines()
gc3 = [l.rstrip() for l in gc3]

all_data.fillna(-1,inplace=True)
for i,pair in enumerate(combinations(cat_cols+num_vars,2)):
    c1 = pair[0]
    c2 = pair[1]
    
    if 'v22' not in [c1,c2] and (c1+c2 in gc or c2+c1 in gc):
        v22_int.append(c1+c2)
        all_data[c1+c2] = all_data[c1].astype(str)+'_'+all_data[c2].astype(str)
print(all_data.shape)
for i,triplet in enumerate(combinations(cat_cols+num_vars,3)):
    c1 = triplet[0]
    c2 = triplet[1]
    c3 = triplet[2]
    fc = [c1,c2,c3]
    fc.sort()
    s=''
    for item in fc:
        s+=item
    if 'v22' not in [c1,c2,c3] and s in gc3:
        v22_int.append(c1+c2+c3)
        all_data[c1+c2+c3] = (all_data[c1].astype(str)+'_'+all_data[c2].astype(str)+'_'
                                    +all_data[c3].astype(str))

#all_data = pd.concat([all_data,
#                      pd.DataFrame('s',index = all_data.index,columns=list(set(v22_int)-set(['ID','target'])))],axis=1)
        
print(all_data.shape)     


    
#train = pd.concat(train_list,axis=1)
#del train_list

train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
y = train.target.values
#train.drop(['target','ID'],axis=1,inplace=True)
train.fillna(-1,inplace=True)
test_idx = test.ID.values
#test.drop(['target','ID'],axis=1,inplace=True)

test.fillna(-1,inplace=True)


#X_train = train.values
skf = StratifiedKFold(y,5,random_state=660)#StratifiedKFold(y,4,shuffle=True,random_state=660)
pred_train = np.zeros(len(y))

baseline = 0.460423 
num_fold = 0

for tr_ind, val_ind in skf:
    #X_tr, X_val = train.iloc[tr_ind].values, train.iloc[val_ind].values
    y_tr, y_val = y[tr_ind], y[val_ind]
    
    X_tr, X_val = train.copy(deep=True).iloc[tr_ind], train.copy(deep=True).iloc[val_ind]
    tr_thr, val_thr = make_avgn(X_tr[v22_int],X_val[v22_int],all_data[v22_int],rs=555)
    np.save('datasets/restmeans_tr'+str(num_fold)+'.npy',tr_thr)
    np.save('datasets/restmeans_val'+str(num_fold)+'.npy',val_thr)
    num_fold+=1
    
print(tr_thr.shape,val_thr.shape)
tr_thr, val_thr = make_avgn(train[v22_int],test[v22_int],all_data[v22_int],rs=555)
np.save('datasets/restmeans_train.npy',tr_thr)
np.save('datasets/restmeans_test.npy',val_thr)


# In[ ]:

from collections import defaultdict
import re

res = np.zeros(X_tr.shape[1])
cols = list(train.drop(v22_int,axis=1).columns)
cols2 = v22_int.copy()
cols2.remove('ID')
cols2.remove('target')
for col in cols2:
    cols.append(col+'_mean_target')
    cols.append(col+'_cnt')

model.dump_model('dump.raw.txt',with_stats=True)

def get_tree_lines(filename):
    with open(filename) as infile:
        line = infile.readline()
        tree_num = int(re.findall('\[(\d+)\]', line)[0])
        tree_lines = []
        for line in infile:
            if line.startswith('booster'):
                yield tree_num, tree_lines
                tree_num = int(re.findall('\[(\d+)\]', line)[0])
                tree_lines = []
            else:
                tree_lines.append(line.strip())
        else:
            yield tree_num, tree_lines

gains = defaultdict(float)
for tree_num, lines in get_tree_lines('dump.raw.txt'):
    tree_nodes = {}
    for line in lines:
        if 'leaf' in line:
            continue
        fnum, gain = re.findall('\[(f\d+)\D.*gain=([\d.]+)', line)[0]
        gain = float(gain)
        gains[fnum] += gain
        
for key in gains.keys():
    ind = int(key[1:])
    res[ind] = gains[key]
    
df = pd.DataFrame(np.c_[np.array(cols),res],columns=['fname','gain'])
df['gain'] = df['gain'].astype(np.float32)
df.sort_values('gain',ascending=False,inplace=True)
#df.to_csv('xgb_fimps.csv',index=False)


# In[ ]:

np.save('datasets/X_tr1.npy',X_tr)
np.save('datasets/X_val1.npy',X_val)
#X_tr.shape


# In[ ]:

#['v3','v24','v31','v66','v71','v74','v91'] - not very important stats
train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
y = train.target.values
train.drop(['target','ID'],axis=1,inplace=True)
train.fillna(-1,inplace=True)
test_idx = test.ID.values
test.drop(['target','ID'],axis=1,inplace=True)

test.fillna(-1,inplace=True)


# In[ ]:

from collections import defaultdict
import re

def get_tree_lines(filename):
    with open(filename) as infile:
        line = infile.readline()
        tree_num = int(re.findall('\[(\d+)\]', line)[0])
        tree_lines = []
        for line in infile:
            if line.startswith('booster'):
                yield tree_num, tree_lines
                tree_num = int(re.findall('\[(\d+)\]', line)[0])
                tree_lines = []
            else:
                tree_lines.append(line.strip())
        else:
            yield tree_num, tree_lines

f_dict = {}
for tree_num, lines in get_tree_lines('dump_combos.txt'):
    for line in lines:
        if 'leaf' in line:
            continue
        fnum = re.findall('\[(f\d+)\D.', line)[0]
        cut  = re.findall('\D.*<([-|\d.]+)', line)[0]
        cut = float(cut)
        f = train.columns[int(fnum[1:])]
        if f not in f_dict.keys():
            f_dict[f] = {}
            if cut not in f_dict[f].keys():
                f_dict[f][cut] = 1
            else:
                f_dict[f][cut] +=1
        else:
            if cut not in f_dict[f].keys():
                f_dict[f][cut] = 1
            else:
                f_dict[f][cut] +=1

f_dict_red = {}
for f in f_dict.keys():
    f_dict_red[f] = set([])
    for cut in f_dict[f].keys():
        if f_dict[f][cut]>1:
            f_dict_red[f].add(cut)


# In[ ]:

#pickle.dump( f_dict_red, open( "f_dict_red.p", "wb" ) )
for key in f_dict_red.keys():
    if key in num_vars:
        print(key,len(f_dict_red[key]))
#f_dict_red['v50']


# In[ ]:

plt.hist(train['v61'],20);


# In[ ]:

import networkx

def findPaths(G,u,n,excludeSet = None):
    if excludeSet == None:
        excludeSet = set([u])
    else:
        excludeSet.add(u)
    if n==0:
        return [[u]]
    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet 
             for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)
    return paths

fcombs2 = {}
fcombs3 = {}


for i in range(2000):
    tree = model.get_dump()[i]
    tree = tree.split()
    G = networkx.Graph()
    node_dict = {}
    for i, text in enumerate(tree):
        if text[0].isdigit():
            if 'leaf' in text:
                continue
            node = int(text.split(':')[0])
            fnum = re.findall('\[(f\d+)\D.', text)[0]
            f = train.columns[int(fnum[1:])]
            node_dict[node] = f
            G.add_node(node)

        else:
            y = int(re.findall('yes=([\d.]+)', text)[0])
            n = int(re.findall('\D.*no=([\d.]+)', text)[0])
            G.add_node(y)
            G.add_edge(node,y)    
            G.add_node(n)
            G.add_edge(node,n)

    '''
    for edge in networkx.edges(G):
        if edge[0] in node_dict.keys() and edge[1] in node_dict.keys():
            f0 = node_dict[edge[0]]
            f1 = node_dict[edge[1]]
            if f0 != f1:
                f_list = [f0,f1]
                f_list.sort()
                s=''
                for item in f_list:
                    s+=item
                if s not in fcombs2.keys():
                    fcombs2[s] = 1
                else:
                    fcombs2[s] += 1
    '''
    allpaths = []
    for node in G:
        allpaths.extend(findPaths(G,node,2))
    for triplet in allpaths:
        if triplet[0] in node_dict.keys() and triplet[1] in node_dict.keys() and triplet[2] in node_dict.keys():
            f0 = node_dict[triplet[0]]
            f1 = node_dict[triplet[1]]
            f2 = node_dict[triplet[2]]
            if f0 != f1 and f0 != f2 and f1 != f2:
                f_list = [f0,f1,f2]
                f_list.sort()
                s=''
                for item in f_list:
                    s+=item
                if s not in fcombs3.keys():
                    fcombs3[s] = 1
                else:
                    fcombs3[s] += 1


# In[ ]:

dd={}
for col in num_vars:
    dd[col] = ['min','max','mean','std','sum','size']
f_dict_red = pickle.load( open( "f_dict_red.p", "rb" ) )
all_data.fillna(-1,inplace=True)
for c in num_vars:
    all_data[c+'raw'] = all_data[c].copy(deep=True)
    bins = np.array(sorted(list(f_dict_red[c])))
    all_data[c] = np.digitize(all_data[c].values,bins)


g_cols = []
v22_int = ['ID','target','v22']
#['min','max','mean','std','sum','size']


with open('goodcombs4.txt','r') as f:
    gc4 = f.readlines()
gc4 = [l.rstrip() for l in gc4]


print(all_data.shape)
for i,quad in enumerate(combinations(cat_cols+num_vars,4)):
    c1 = quad[0]
    c2 = quad[1]
    c3 = quad[2]
    c4 = quad[3]
    fc = [c1,c2,c3,c4]
    fc.sort()
    s=''
    for item in fc:
        s+=item
    if 'v22' not in [c1,c2,c3,c4] and s in gc4:
        v22_int.append(c1+c2+c3+c4+'v22')
        all_data[c1+c2+c3+c4+'v22'] = (all_data[c1].astype(str)+'_'+all_data[c2].astype(str)+'_'
                                    +all_data[c3].astype(str)+'_'+all_data[c4].astype(str)+'_'
                                       +all_data['v22'].astype(str))

#all_data = pd.concat([all_data,
#                      pd.DataFrame('s',index = all_data.index,columns=list(set(v22_int)-set(['ID','target'])))],axis=1)
        
print(all_data.shape)     
all_data.drop(num_vars,axis=1,inplace=True)
print(all_data.shape)

    
train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
y = train.target.values
#train.drop(['target','ID'],axis=1,inplace=True)
train.fillna(-1,inplace=True)
test_idx = test.ID.values
#test.drop(['target','ID'],axis=1,inplace=True)

test.fillna(-1,inplace=True)



tr_thr, val_thr = make_avg(train[v22_int],test[v22_int],all_data[v22_int],rs=555)

print(tr_thr.shape,val_thr.shape)
np.save('datasets/v22means2_train.npy',tr_thr)
np.save('datasets/v22means2_test.npy',val_thr)


# In[ ]:



