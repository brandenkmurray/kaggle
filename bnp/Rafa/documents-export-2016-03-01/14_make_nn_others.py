# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.grid_search import ParameterGrid
from sklearn.preprocessing import StandardScaler

################################# CONFIG ######################################

data_path = "C:/Users/Rafael/Documents/data/bnp/data/"
#data_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/"

learners = [
    {
        "learner": NearestNeighbors,
        "dirs": ["raw/"],
        "param_grid": ParameterGrid({
            "leaf_size": [1024],
            "p": [1, 2]
        })
    }
]

################################# SCRIPT ######################################

logging.basicConfig(filename=data_path + os.path.basename(__file__).split(".")[0] + ".log",
                level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

try:
    candidate = learners[0]
    logging.info("*******************************************************")
    
    for tipo, data_dir in zip(["nan"], candidate["dirs"]):
        X_train = pd.read_csv(data_path + data_dir + "train.csv", index_col=0)
        y_train = pd.read_pickle(data_path + "raw_filled/train.p")
        y_train = y_train.target
        idx_train = y_train.index
        y_train = y_train.values.astype("int32", order="C")
        if "target" in X_train.columns:
            X_train = X_train.drop("target", axis=1)
        
        X_test = pd.read_csv(data_path + data_dir + "test.csv", index_col=0)
        idx_test = X_test.index
        
        if tipo == "nan":
            X_train = X_train.isnull()
            X_test = X_test.isnull()
            X_train = X_train.ix[idx_train].values.astype("int8", order="C")
            X_test = X_test.values.astype("int8", order="C")
        else:
            X_train = X_train.ix[idx_train].values.astype("float32", order="C")
            X_test = X_test.values.astype("float32", order="C")
            
        knn = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
        for params in candidate["param_grid"]:
            logging.info("----------------------------------------------------------")
            learner = candidate["learner"]
            learner_name = learner.__name__ + "_" + str(params["p"])
            
            train_probs_path = data_path +  "NN_" + str(params["p"]) + "_" + tipo + "/train.p"
            test_probs_path = data_path + "NN_" + str(params["p"]) + "_" + tipo + "/test.p"
    
            logging.info("Entrenando...")
    
    
            learner = learner(**params)
            learner.fit(X_train)
    
            cols = ["NN_" + str(i) for i in knn]
            cols = cols + ["c1_1", "c2_1", "c1_0",
                           "c2_0", "c1_ratio", "c2_ratio"]
            cols = [c + "_" + str(params["p"]) for c in cols]
    
            logging.info("generando datos para train...")
            train_probs = pd.DataFrame([], columns=cols, index=idx_train)
            for i, x in enumerate(X_train):
                nn = learner.kneighbors(x.reshape(1, -1), max(knn) + 1)
                idx = idx_train[i]
                dists = nn[0][0, 1:]
                nn_idx = nn[1][0, 1:]
                classes = y_train[nn_idx]
                for k in knn:
                    train_probs.ix[idx, "NN_" + str(k) + str(params["p"])] = classes[:k].sum() / classes[:k].shape[0]
                c_0 = dists[classes == 0][:2]
                c_1 = dists[classes == 1][:2]
                train_probs.ix[idx, "c1_1"] = c_1[0]
                train_probs.ix[idx, "c2_1"] = c_1.sum()
                train_probs.ix[idx, "c1_0"] = c_0[0]
                train_probs.ix[idx, "c2_0"] = c_0.sum()
                train_probs.ix[idx, "c1_ratio"] = c_1[0] / c_0[0]
                train_probs.ix[idx, "c2_ratio"] = c_1.sum() / c_0.sum()
                if i % 1000 == 0:
                    logging.info("generados {1:.2f}%".format(i, i / X_train.shape[0] * 100))
    
            logging.info("generando datos para test...")
            test_probs = pd.DataFrame([], columns=cols, index=idx_test)
            for i, x in enumerate(X_test):
                nn = learner.kneighbors(x.reshape(1, -1), max(knn))
                idx = idx_test[i]
                dists = nn[0][0]
                nn_idx = nn[1][0]
                classes = y_train[nn_idx]
                for k in knn:
                    test_probs.ix[idx, "NN_" + str(k) + str(params["p"])] = classes[:k].sum() / classes[:k].shape[0]
                c_0 = dists[classes == 0][:2]
                c_1 = dists[classes == 1][:2]
                test_probs.ix[idx, "c1_1"] = c_1[0]
                test_probs.ix[idx, "c2_1"] = c_1.sum()
                test_probs.ix[idx, "c1_0"] = c_0[0]
                test_probs.ix[idx, "c2_0"] = c_0.sum()
                test_probs.ix[idx, "c1_ratio"] = c_1[0] / c_0[0]
                test_probs.ix[idx, "c2_ratio"] = c_1.sum() / c_0.sum()
                if i % 1000 == 0:
                    logging.info("generados {1:.2f}%".format(i, i / X_test.shape[0] * 100))
    
            logging.info("Guardando resultados...")
            train_probs.to_pickle(train_probs_path)
            test_probs.to_pickle(test_probs_path)
except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")
