# -*- coding: utf-8 -*-

import sys
import os
import gc
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(".."); from Perceptron import BinaryMLP
from xgboost import XGBClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

################################# CONFIG ######################################

#data_path = "C:/Users/Rafael/Documents/data/bnp/"
data_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/"

learners = [
#    {
#        "learner": ExtraTreesClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [500],
#            "criterion": ["entropy"],
#            "max_features": ["auto"],
#            "max_depth": [35, None],
#            "min_weight_fraction_leaf": [0.0],
#            "bootstrap": [False, True],
#            "class_weight": [None, "balanced", "balanced_subsample"],
#            "n_jobs": [-1]
#        })
#    },
#    {
#        "learner": GradientBoostingClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [300],
#            "learning_rate": [0.1],
#            "max_features": ["auto"],
#            "max_depth": [3],
#            "subsample": [0.5],
#            "min_weight_fraction_leaf": [0.0],
#            "presort": [True]
#        })
#    },
#    {
#        "learner": GradientBoostingClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [160],
#            "learning_rate": [0.1],
#            "max_features": ["auto"],
#            "max_depth": [5],
#            "subsample": [0.5],
#            "min_weight_fraction_leaf": [0.0001],
#            "presort": [True]
#        })
#    },
#    {
#        "learner": GradientBoostingClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [65],
#            "learning_rate": [0.1],
#            "max_features": ["auto"],
#            "max_depth": [8, 10, 15, 25],
#            "subsample": [0.5],
#            "min_weight_fraction_leaf": [0.01],
#            "presort": [True]
#        })
#    },
#    {
#        "learner": XGBClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [300],
#            "learning_rate": [0.1],
#            "colsample_bytree": [0.65],
#            "colsample_bylevel": [1],
#            "max_depth": [3, 5],
#            "gamma": [0.0],
#            "subsample": [1],
#            "min_child_weight": [0],
#            "base_score": [0.7612]
#        })
#    },
#    {
#        "learner": XGBClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [75, 150, 500],
#            "learning_rate": [0.1],
#            "colsample_bytree": [0.65],
#            "colsample_bylevel": [1],
#            "max_depth": [7, 8, 10],
#            "gamma": [0.0],
#            "subsample": [1],
#            "min_child_weight": [0, 250],
#            "base_score": [0.7612]
#            })
#    },
#    {
#        "learner": XGBClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [100],
#            "learning_rate": [0.1],
#            "colsample_bytree": [0.65],
#            "colsample_bylevel": [1],
#            "max_depth": [15, 25, 100],
#            "gamma": [0.0],
#            "subsample": [1],
#            "min_child_weight": [100],
#            "base_score": [0.7612]
#        })
#    },
#    {
#        "learner": RandomForestClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [500],
#            "criterion": ["entropy"],
#            "max_features": ["auto"],
#            "max_depth": [20, None],
#            "min_weight_fraction_leaf": [0.0, 0.001, 0.0001],
#            "bootstrap": [False],
#            "class_weight": [None, "balanced", "balanced_subsample"],
#            "n_jobs": [-1]
#        })
#    },
#    {
#        "learner": RandomForestClassifier,
#        "dir": "raw_filled/",
#        "param_grid": ParameterGrid({
#            "n_estimators": [500],
#            "criterion": ["entropy"],
#            "max_features": ["auto"],
#            "max_depth": [3],
#            "min_weight_fraction_leaf": [0.01],
#            "bootstrap": [True],
#            "class_weight": ["balanced"],
#            "n_jobs": [-1]
#        })
#    },
    {
        "learner": LogisticRegression,
        "dir": "raw_filled/",
        "scale": True,
        "param_grid": ParameterGrid({
            "C": [1000, 100, 10, 1],
            "penalty": ["l2"],
            "solver": ["sag"],
            "class_weight": [None, "balanced"]
        })}
#    },
#    {
#        "learner": BinaryMLP,
#        "dir": "raw_filled/",
#        "scale": True,
#        "param_grid": ParameterGrid({
#            "hidden": [(100, 0), (100, 50)],
#            "drop": [(0.1, 0.1)],
#            "activations": [("relu", "sigmoid"), ("tanh", "sigmoid")],
#            "patience": [20],
#            "save_path": [data_path + "NN.hdf5"]
#        })
#    }
]

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=data_path + file_name + ".log",
                level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

resultados_path = data_path + "probs_stage_1/" + file_name + "_resultados.xlsx"
train_probs_path = data_path + "probs_stage_1/train_probs.p"
test_probs_path = data_path + "probs_stage_1/test_probs.p"
test_probs_avg_path = data_path + "probs_stage_1/test_probs_avg.p"

folds = np.load("folds.npy")

try:
    if os.path.exists(resultados_path):
        resultados = pd.read_excel(resultados_path)
        resultados["params"] = [eval(d) for d in resultados.params]
        train_probs = pd.read_pickle(train_probs_path)
        test_probs = pd.read_pickle(test_probs_path)
        test_probs_avg = pd.read_pickle(test_probs_path)
    else:
        resultados = pd.DataFrame([])
        train_probs = pd.DataFrame([])
        test_probs = pd.DataFrame([])
        test_probs_avg = pd.DataFrame([])

    for candidate in learners:
        logging.info("*******************************************************")
        logging.info("Leyendo datos para nuevo candidato")

        X_train = pd.read_pickle(data_path + candidate["dir"] + "train.p")
        y_train = X_train.target.values.astype("int32", order="C")
        idx_train = X_train.index
        X_train = X_train.drop("target", axis=1).values.astype("float32", order="C")

        X_test = pd.read_pickle(data_path + candidate["dir"] + "test.p")
        idx_test = X_test.index
        X_test = X_test.values.astype("float32", order="C")

        if candidate.get("scale", False):
            logging.info("Normalizando datos...")
            scaler = StandardScaler()
            scaler.fit(np.r_[X_train, X_test])
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        for params in candidate["param_grid"]:
            logging.info("----------------------------------------------------------")
            learner = candidate["learner"]
            learner_kind = learner.__name__
            cand_params = resultados.filter(regex=learner_kind, axis=0)
            l_nb = len(cand_params.index)
            learner_name = learner_kind + "_" + str(l_nb)



            if ("params" in resultados.columns) and np.any(cand_params.params == params):
                logging.info("SALTENADO learner ya entrenado: " + learner_name)
                logging.info("parametros:" + str(params))
                continue
            else:
                logging.info("Entrenando learner: " + learner_name)
                logging.info("parametros:" + str(params))

            logging.info("Entrenando...")

            l_probs = []
            idx_probs = []
            l_test_probs_avg = []
            i = 0
            for train_index, test_index in folds:
                i += 1
                logging.info("Entrenando fold " + str(i))
                cur_learner = learner(**params)
                cur_learner.fit(X_train[train_index], y_train[train_index])
                l_probs.append(cur_learner.predict_proba(X_train[test_index])[:, -1])
                l_test_probs_avg.append(pd.Series(cur_learner.predict_proba(X_test)[:, -1],
                                                  index=idx_test))
                del cur_learner
                gc.collect()
                fold_tes = log_loss(y_train[test_index], l_probs[-1])
                logging.info("Resultado en fold {0} fue de {1} ".format(i, fold_tes))
                idx_probs.append(idx_train[test_index])

            l_test_probs_avg = pd.concat(l_test_probs_avg, axis=1).mean(axis=1)
            l_test_probs_avg.name = learner_name
            test_probs_avg = pd.concat([test_probs_avg, l_test_probs_avg], axis=1)

            logging.info("Calculando resultados")

            l_probs = pd.Series(np.concatenate(l_probs), index=np.concatenate(idx_probs))
            l_probs.name = learner_name

            train_probs = pd.concat([train_probs, l_probs], axis=1)

            resultado = log_loss(y_train, l_probs.ix[idx_train].values)
            logging.info("resultaodo fue de: " + str(resultado))

            logging.info("Entrenando para test...")
            cur_learner = learner(**params)
            cur_learner.fit(X_train, y_train)

            l_probs = pd.Series(cur_learner.predict_proba(X_test)[:, -1],
                                index=idx_test)
            del cur_learner
            gc.collect()

            l_probs.name = learner_name

            test_probs = pd.concat([test_probs, l_probs], axis=1)

            logging.info("Guardando resultados...")

            cols = ["resultado", "params"]
            vals = [resultado, params]
            for p in params:
                cols.append("param_" + p)
                vals.append(params[p])
            vals = np.asarray(vals)
            vals = vals.reshape((1, vals.shape[0]))
            resultados = resultados.append(pd.DataFrame(vals, columns=cols, index=[learner_name]))
            resultados.to_excel(resultados_path)
            train_probs.to_pickle(train_probs_path)
            test_probs_avg.to_pickle(test_probs_avg_path)
            test_probs.to_pickle(test_probs_path)
except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")
