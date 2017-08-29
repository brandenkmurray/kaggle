# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

base_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/"
#base_path = "C:/Users/Rafael/Documents/data/bnp/"

data_path = base_path + "raw/"
filled_path = base_path + "raw_filled/"

train = pd.read_csv(data_path + "train.csv", index_col=0)
test = pd.read_csv(data_path + "test.csv", index_col=0)

train["nan_count"] = train.isnull().sum(axis=1).astype("int64")
test["nan_count"] = test.isnull().sum(axis=1).astype("int64")

filled_train = train.copy()
filled_test = test.copy()

base = np.ones((train.shape[0], 2))
base_probs = train.target.value_counts() / train.shape[0]
base = base * [base_probs.ix[0], base_probs.ix[1]]
base = log_loss(train.target, base)

gen_prop = train.target.mean()

for c in train.drop(["target"], axis=1):
    if train[c].dtype == "int64":
        ct = pd.crosstab(train[c], train.target).apply(lambda x: x / x.sum(), axis=1).iloc[:, 1]
        rel_cats = train[c].value_counts()
        irel_idx = rel_cats[rel_cats < 200].index
        ct.ix[irel_idx] = gen_prop

        filled_train[c + "_bayes"] = filled_train[c].map(ct).fillna(gen_prop)
        filled_test[c + "_bayes"] = test[c].fillna("N/A").map(ct).fillna(gen_prop)
    elif (train[c].dtype == "float64"):
        if (train[c].isnull().sum() == 0):
            continue
        cur_res = {}
        for met in [-1, 21]:
            l = DecisionTreeClassifier(criterion="entropy", max_depth=1)
            x_train = train[c].fillna(met).reshape(-1, 1)
            probs = l.fit(x_train, train.target).predict_proba(x_train)
            cur_res[met]  = base - log_loss(train.target, probs)
        if cur_res[21] > cur_res[-1]:
            filled_train[c] = filled_train[c].fillna(21)
            filled_test[c] = filled_test[c].fillna(21)
        else:
            filled_train[c] = filled_train[c].fillna(-1)
            filled_test[c] = filled_test[c].fillna(-1)
    else:
        x_train = train[c].fillna("N/A")
        ct = pd.crosstab(x_train, train.target)

        rel_cats = x_train.value_counts()
        irel_idx = rel_cats[rel_cats < 200].index
        ct = ct.apply(lambda x: x / x.sum(), axis=1).iloc[:, 1]

        ct.ix[irel_idx] = gen_prop
        if "N/A" in ct.index:
            filler = ct.ix["N/A"]
            ct.ix["N/A"] = -9e10
        filled_train[c] = x_train.map(ct).fillna(gen_prop)
        filled_train[c] = filled_train[c].replace(-9e10, filler)
        filled_test[c] = test[c].fillna("N/A").map(ct).fillna(gen_prop)
        filled_test[c] = filled_test[c].replace(-9e10, filler)

    print("done", c)

filled_train.to_pickle(filled_path + "train.p")
filled_test.to_pickle(filled_path + "test.p")


folds = [[train_index, test_index] for train_index, test_index
         in StratifiedKFold(train.target.values, n_folds=5, shuffle=True)]
folds = np.asarray(folds)
np.save("folds", folds)
