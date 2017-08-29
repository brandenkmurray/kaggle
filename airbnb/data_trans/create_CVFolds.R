library(xgboost)
library(data.table)
library(readr)
library(caret)

setwd("/home/branden/Documents/kaggle/airbnb")
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v1.csv"))

set.seed(2016)
cvFolds <- data.frame(foldIndex=createFolds(ts1Trans$class[ts1Trans$filter==0], k=5, list=FALSE))
set.seed(2016)
cvFoldsList <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=5, list=TRUE, returnTrain=FALSE)
set.seed(2016)
cvFoldsTrainList <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=5, list=TRUE, returnTrain=TRUE)

write_csv(cvFolds, "./data_trans/cvFolds.csv")
save(cvFoldsList, file="./data_trans/cvFoldsList.rda")
save(cvFoldsTrainList, file="./data_trans/cvFoldsTrainList.rda")

set.seed(2016)
cvFolds_lay2 <- data.frame(foldIndex=createFolds(ts1Trans$class[ts1Trans$filter==0], k=4, list=FALSE))
set.seed(2016)
cvFoldsList_lay2 <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=4, list=TRUE, returnTrain=FALSE)

write_csv(cvFolds_lay2, "./data_trans/cvFolds_lay2.csv")
save(cvFoldsList_lay2, file="./data_trans/cvFoldsList_lay2.rda")

set.seed(2016)
cvFolds_lay2_k6 <- data.frame(foldIndex=createFolds(ts1Trans$class[ts1Trans$filter==0], k=6, list=FALSE))
set.seed(2016)
cvFoldsList_lay2_k6 <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=6, list=TRUE, returnTrain=FALSE)

write_csv(cvFolds_lay2_k6, "./data_trans/cvFolds_lay2_k6.csv")
save(cvFoldsList_lay2_k6, file="./data_trans/cvFoldsList_lay2_k6.rda")
