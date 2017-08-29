library(readr)
library(data.table)
library(caret)
setwd("/media/branden/SSHD1/kaggle/expedia")


train <- fread("./train.csv")
set.seed(2016)
cvFoldsList <- createFolds(train$hotel_cluster, k=5, list=TRUE, returnTrain=FALSE)
# TrainList for use with Caret
set.seed(2016)
cvFoldsTrainList <- createFolds(train$hotel_cluster, k=5, list=TRUE, returnTrain=TRUE)
set.seed(2016)
cvFolds <- data.frame(foldIndex=createFolds(train$hotel_cluster, k=5, list=FALSE))

save(cvFoldsList, file="./data_trans/cvFoldsList.rda")
save(cvFoldsTrainList, file="./data_trans/cvFoldsTrainList.rda")
write_csv(cvFolds, "./data_trans/cvFolds.csv")

set.seed(1234)
cvFoldsList_lay2 <- createFolds(train$hotel_cluster, k=6, list=TRUE, returnTrain=FALSE)
set.seed(1234)
cvFoldsTrainList_lay2 <- createFolds(train$hotel_cluster, k=6, list=TRUE, returnTrain=TRUE)
set.seed(1234)
cvFolds_lay2 <- data.frame(foldIndex=createFolds(train$hotel_cluster, k=6, list=FALSE))

save(cvFoldsList_lay2, file="./data_trans/cvFoldsList_lay2.rda")
save(cvFoldsTrainList_lay2, file="./data_trans/cvFoldsTrainList_lay2.rda")
write_csv(cvFolds_lay2, "./data_trans/cvFolds_lay2.csv")
