# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans

################################# CONFIG ######################################

#base_path = "C:/Users/Rafael/Documents/data/bnp/"
base_path = "C:/Users/rcrescenzi/Documents/Personal/data/bnp/"
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

    X_train = pd.read_pickle(target_path + "raw_filled/train.p")
    X_test = pd.read_pickle(target_path + "raw_filled/test.p")
    y_train = X_train.target

#    Tfidf_X_train = X_train.drop("target", axis=1).replace(-1, 0).clip(0, 21)
#    Tfidf_X_test = X_test.replace(-1, 0).clip(0, 21)
#
#    trans = TfidfTransformer().fit((pd.concat([Tfidf_X_train, Tfidf_X_test])))
#
#    Tfidf_X_train = pd.DataFrame(trans.transform(Tfidf_X_train).toarray(),
#                                 index=Tfidf_X_train.index, columns=Tfidf_X_train.columns)
#    Tfidf_X_test = pd.DataFrame(trans.transform(Tfidf_X_test).toarray(),
#                                index=Tfidf_X_test.index, columns=Tfidf_X_test.columns)
#
#    Tfidf_X_train.to_pickle(target_path + "tfidf/train.p")
#    Tfidf_X_test.to_pickle(target_path + "tfidf/test.p")
#
#    for tipo, train, test in [("raw_filled", X_train.drop("target", axis=1), X_test),
#                              ("tfidf", Tfidf_X_train, Tfidf_X_test)]:
#
#        KM_train = []
#        KM_test = []
#
#        for k in [2, 5, 10]:
#
#            trans = MiniBatchKMeans(k).fit((pd.concat([train, test])))
#
#            KM_train.append(pd.DataFrame(trans.transform(train), index=train.index,
#                                         columns=["KM_" + str(k) + "_" + str(i + 1)
#                                                  for i  in range(k)]))
#            KM_test.append(pd.DataFrame(trans.transform(test), index=test.index,
#                                        columns=["KM_" + str(k) + "_" + str(i + 1)
#                                                 for i  in range(k)]))
#
#        pd.concat(KM_train, axis=1).to_pickle(target_path + "kmeans_" + tipo + "/train.p")
#        pd.concat(KM_test, axis=1).to_pickle(target_path + "kmeans_" + tipo + "/test.p")

    X_train_p = X_train.ix[(y_train[y_train == 1]).index].drop("target", axis=1)
    X_train_n = X_train.ix[(y_train[y_train == 0]).index].drop("target", axis=1)

    KM_train = []
    KM_test = []

    for tipo, train in zip(["pos", "neg"], [X_train_p, X_train_n]):
        for k in [1, 2, 4, 6]:
            trans = MiniBatchKMeans(k).fit(train)

            KM_train.append(pd.DataFrame(trans.transform(X_train.drop("target", axis=1)), index=X_train.index,
                                         columns=["KM_" + str(k) + "_" + tipo + "_" + str(i)
                                                  for i  in range(k)]))
            KM_test.append(pd.DataFrame(trans.transform(X_test), index=X_test.index,
                                        columns=["KM_" + str(k) + "_"  + tipo + "_" + str(i)
                                                 for i  in range(k)]))

    pd.concat(KM_train, axis=1).to_pickle(target_path + "kmeans_pos_neg/train.p")
    pd.concat(KM_test, axis=1).to_pickle(target_path + "kmeans_pos_neg/test.p")

except Exception as e:
    logging.exception("***** Se ha encontrado un error *****")
finally:
    import winsound
    winsound.Beep(1000, 600)
    logging.info("***** TERMINADO *****")

