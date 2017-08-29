# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:02:07 2016

@author: branden
"""

import pandas as pd 
import numpy as np 
import xgboost as xgb



# Set XGBOOST parameters
params = {}
params["objective"] = "reg:logistic"     
params["eta"] = 0.01
params["min_child_weight"] = 1
params["subsample"] = 0.8
params["colsample_bytree"] = 0.4
params["silent"] = 1
params["max_depth"] = 7
params["eval_metric"] = "logloss"
params["max_delta_step"] = 1



# global variables
columns_to_drop = ['ID', 'target', 'pred0','dummy','filter']
xgb_num_rounds = 7000
num_classes = 2
eta_list = [0.05] * 50 + [0.04] * 50 + [0.03] * 50 + [0.02] * 50 + [0.01] * 9800


ts1Trans = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/ts2Trans_v24.csv")
cvFolds = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/data_trans/cvFolds10.csv")
folds = np.max(cvFolds).astype(np.float32)[0]
  

cv_by_hand = [(np.where(cvFolds != fold)[0], np.where(cvFolds == fold)[0])
               for fold in np.unique(cvFolds)]

xgtest = xgb.DMatrix(ts1Trans[ts1Trans['filter']==2].drop(columns_to_drop, axis=1, errors='ignore'))

cvPreds = np.zeros((ts1Trans[ts1Trans['filter']==0].shape[0], ))
testPreds = np.zeros((ts1Trans[ts1Trans['filter']==2].shape[0], ))
for fold in xrange(0,np.max(cvFolds)):
    print "Fold", fold
    xgtrain = xgb.DMatrix(ts1Trans[ts1Trans['filter']==0].iloc[cv_by_hand[fold][0],:].drop(columns_to_drop, axis=1, errors='ignore'), ts1Trans.loc[ts1Trans['filter']==0,'target'][cv_by_hand[fold][0]].values)
    xgval = xgb.DMatrix(ts1Trans[ts1Trans['filter']==0].iloc[cv_by_hand[fold][1],:].drop(columns_to_drop, axis=1, errors='ignore'), ts1Trans.loc[ts1Trans['filter']==0,'target'][cv_by_hand[fold][1]].values)
    
    watchlist  = [(xgtrain,'train'), (xgval,'eval')]
    np.random.seed(56)    
    model = xgb.train(params, xgtrain, num_boost_round=len(eta_list), evals=watchlist, learning_rates=eta_list, early_stopping_rounds=200,verbose_eval=50, maximize=False)
#    model = xgb.train(params, xgtrain, num_boost_round=5, evals=watchlist, early_stopping_rounds=200,verbose_eval=20, maximize=False)    
    cvPreds[cv_by_hand[fold][1]] += model.predict(xgval, ntree_limit=model.best_iteration)
    
    testPreds += model.predict(xgtest, ntree_limit=model.best_iteration)

testPreds_adj = testPreds/folds        

testPreds_adj = testPreds/3

samp = pd.read_csv("/media/branden/SSHD1/kaggle/bnp/sample_submission.csv") 

cvPreds_frame = pd.DataFrame(cvPreds, columns=samp.columns[1:])
#s1nn = s1nn.reset_index(drop=True)
cvPreds_frame.insert(0, 'ID', ts1Trans.loc[ts1Trans['filter']==0,'ID'])
cvPreds_frame.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/cvPreds/cvPreds_xgb36.csv", index=False)

testPreds_frame = pd.DataFrame(testPreds_adj, columns=samp.columns[1:])
testPreds_frame = testPreds_frame.set_index(ts1Trans.loc[ts1Trans['filter']==2,'ID'].index)
testPreds_frame.insert(0, 'ID', ts1Trans.loc[ts1Trans['filter']==2,'ID'].astype(np.int32))
testPreds_frame.to_csv("/media/branden/SSHD1/kaggle/bnp/stack_models/testPreds/testPreds_xgb36.csv", index=False)

