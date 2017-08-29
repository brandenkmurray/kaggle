# -*- coding: utf-8 -*-

import sys
import os
import gc
import logging

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.grid_search import ParameterSampler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

def read_data(data_path, tipo):
    f_tipo = [a.split(".")[-1] for a in os.listdir(data_path)]
    if "csv" in f_tipo:
        return pd.read_csv(data_path + "/" + tipo + ".csv", index_col=0)
    else:
        return pd.read_pickle(data_path + "/" + tipo + ".p")

################################# CONFIG ######################################

#base_path = "C:/Users/Rafael/Documents/data/bnp/data/"
base_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/data/"
target_path = base_path + "probs_stage_2_XGB_bag/"

learners = [
    {
        "learner": XGBClassifier,
        "dir": "stacked_all/probs_stage_2/",
        "param_grid": ParameterSampler({
            "n_estimators": [50000],
            "learning_rate": [0.01],
            "colsample_bytree": [0.4],
            "colsample_bylevel": [1],
            "max_depth": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            "gamma": [0, 0, 0, 0, 0, 0, 0],
            "subsample": [1],
            "min_child_weight": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            "base_score": [0.7612]
            }, 200)
    }
]

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=target_path +  file_name + ".log",
                level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

resultados_path = target_path + "resultados.xlsx"
train_probs_path = target_path + "train_probs.p"
test_probs_path = target_path + "test_probs.p"
test_probs_avg_path = target_path + "test_probs_avg.p"

folds = np.load("folds.npy")

try:
    if os.path.exists(resultados_path):
        resultados = pd.read_excel(resultados_path)
        resultados["params"] = [eval(d) for d in resultados.params]
        train_probs = pd.read_pickle(train_probs_path)
        test_probs = pd.read_pickle(test_probs_path)
        test_probs_avg = pd.read_pickle(test_probs_avg_path)
    else:
        resultados = pd.DataFrame([])
        train_probs = pd.DataFrame([])
        test_probs = pd.DataFrame([])
        test_probs_avg = pd.DataFrame([])

    for candidate in learners:
        logging.info("*******************************************************")
        logging.info("Leyendo datos para nuevo candidato")

        data_dirs = ["probs_stage_1", "kmeans_nulls", "kmeans_raw_filled",
                     "kmeans_tfidf", "summary", "KNN_p1_raw_filled", "KNN_p2_raw_filled",
                     "xgb1", "xgb2", "xgb3", "xgb4", "xgb5_best", "xgb6",
                     "xgb7_best", "xgb8", "xgb9_best", "xgb13"]

        X_train = pd.concat([read_data(base_path + d, "train") for d in data_dirs], axis=1)
        X_train = X_train.dropna(axis=1)
        y_train = pd.read_pickle(base_path + "raw_filled/train.p")
        y_train = y_train.target
        idx_train = y_train.index
        y_train = y_train.values.astype("int32", order="C")
        if "target" in X_train.columns:
            X_train.drop("target", axis=1, inplace=True)
        X_train = X_train.ix[idx_train].values.astype("float32", order="C")

        X_test = pd.concat([read_data(base_path + d, "test") for d in data_dirs], axis=1)
        idx_test = X_test.index
        X_test = X_test.dropna(axis=1)
        X_test = X_test.ix[idx_test].values.astype("float32", order="C")

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

#            if ("params" in resultados.columns) and np.any(cand_params.params == params):
#                logging.info("SALTENADO learner ya entrenado: " + learner_name)
#                logging.info("parametros:" + str(params))
#                continue
#            else:
#                logging.info("Entrenando learner: " + learner_name)
#                logging.info("parametros:" + str(params))
            logging.info("Entrenando learner: " + learner_name)
            logging.info("parametros:" + str(params))
            logging.info("Entrenando...")

            l_probs = []
            idx_probs = []
            l_test_probs_avg = []
            i = 0
            for train_index, test_index in StratifiedKFold(y_train, 10, shuffle=True):
                i += 1
                logging.info("Entrenando fold " + str(i))
                cur_learner = learner(**params)
                if learner_kind in ["XGBClassifier", "BinaryMLP"]:
                    cur_learner.fit(X_train[train_index], y_train[train_index],
                                    eval_set=[(X_train[test_index], y_train[test_index])],
                                    early_stopping_rounds=20, eval_metric="logloss", verbose=True)
                    if learner_kind == "XGBClassifier":
                        n_trees = cur_learner.best_iteration
                        l_probs.append(cur_learner.predict_proba(X_train[test_index], ntree_limit=n_trees)[:, -1])
                        l_test_probs_avg.append(pd.Series(cur_learner.predict_proba(X_test, ntree_limit=n_trees)[:, -1],
                                                          index=idx_test))
                    else:
                        l_probs.append(cur_learner.predict_proba(X_train[test_index])[:, -1])
                        l_test_probs_avg.append(pd.Series(cur_learner.predict_proba(X_test)[:, -1],
                                                          index=idx_test))
                else:
                    cur_learner.fit(X_train[train_index], y_train[train_index])
                    l_probs.append(cur_learner.predict_proba(X_train[test_index])[:, -1])
                    l_test_probs_avg.append(pd.Series(cur_learner.predict_proba(X_test)[:, -1],
                                                      index=idx_test))
                gc.collect()
                fold_tes = log_loss(y_train[test_index], l_probs[-1])
                logging.info("Resultado en fold {0} fue de {1}".format(i, fold_tes))
                idx_probs.append(idx_train[test_index])
                del cur_learner

            l_test_probs_avg = pd.concat(l_test_probs_avg, axis=1).mean(axis=1)
            l_test_probs_avg.name = learner_name
            test_probs_avg = pd.concat([test_probs_avg, l_test_probs_avg], axis=1)

            logging.info("Calculando resultados")

            l_probs = pd.Series(np.concatenate(l_probs), index=np.concatenate(idx_probs))
            l_probs.name = learner_name

            train_probs = pd.concat([train_probs, l_probs], axis=1)

            resultado = log_loss(y_train, l_probs.ix[idx_train].values)
            logging.info("resultaodo fue de: " + str(resultado))

#            logging.info("Entrenando para test...")
#            cur_learner = learner(**params)
#
#            if learner_kind in ["XGBClassifier", "BinaryMLP"]:
#                cur_learner.fit(X_train[train_index], y_train[train_index],
#                                eval_set=[(X_train[test_index], y_train[test_index])],
#                                early_stopping_rounds=20, eval_metric="logloss", verbose=True)
#                if learner_kind == "XGBClassifier":
#                    n_trees = cur_learner.best_iteration
#                    l_probs = pd.Series(cur_learner.predict_proba(X_test, ntree_limit=n_trees)[:, -1],
#                                        index=idx_test)
#                else:
#                    l_probs = pd.Series(cur_learner.predict_proba(X_test)[:, -1],
#                                        index=idx_test)
#            else:
#                cur_learner.fit(X_train[train_index], y_train[train_index])
#                l_probs = pd.Series(cur_learner.predict_proba(X_test)[:, -1],
#                                    index=idx_test)
#
#
#            del cur_learner
#            gc.collect()
#
#            l_probs.name = learner_name
#
#            test_probs = pd.concat([test_probs, l_probs], axis=1)

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
