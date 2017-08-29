# -*- coding: utf-8 -*-

import os
import logging

import pandas as pd

from sklearn.cluster import MiniBatchKMeans

################################# CONFIG ######################################

base_path = "C:/Users/Rafael/Documents/data/bnp/"
#base_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/"
target_path = base_path + "data/"

################################# SCRIPT ######################################

file_name = os.path.basename(__file__).split(".")[0]

logging.basicConfig(filename=target_path + file_name + ".log",
                level=logging.DEBUG, format='%(asctime)s -- %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())

logging.info("***** Iniciando rutina de busqueda de hiperparametros *****")

try:
    logging.info("*******************************************************")
    logging.info("Leyendo datos para nuevo candidato")

    X_train = pd.read_csv(target_path + "raw/train.csv", index_col=0)
    X_test = pd.read_csv(target_path + "raw/test.csv", index_col=0)
    y_train = X_train.target
    X_train = X_train.drop("target", axis=1)

    dtypes = X_train.dtypes
    float_cols = dtypes[dtypes=="float64"].index
    int_cols = dtypes[dtypes=="int64"].index

    summaries_train = []
    summaries_test = []
    for tipo, cols in [("float", float_cols), ("int", int_cols)]:
        for func in ("max", "min", "mean", "std"):
            temp = getattr(X_train[cols], func)(axis=1)
            temp.name = tipo + "_" + func
            summaries_train.append(temp)

            temp = getattr(X_test[cols], func)(axis=1)
            temp.name = tipo + "_" + func
            summaries_test.append(temp)

    pd.concat(summaries_train, axis=1).to_pickle(target_path + "summary/train.p")
    pd.concat(summaries_test, axis=1).to_pickle(target_path + "summary/test.p")

    nuls_train = X_train.isnull()
    nuls_test = X_test.isnull()

    trans = MiniBatchKMeans(10).fit((pd.concat([nuls_train, nuls_test])))

    nuls_train = pd.DataFrame(trans.transform(nuls_train), index=nuls_train.index,
                                      columns=["KM_10_" + str(i + 1)
                                              for i  in range(10)])
    nuls_test = pd.DataFrame(trans.transform(nuls_test), index=nuls_test.index,
                                      columns=["KM_10_" + str(i + 1)
                                              for i  in range(10)])

    nuls_train.to_pickle(target_path + "kmeans_nulls/train.p")
    nuls_test.to_pickle(target_path + "kmeans_nulls/test.p")

except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")
