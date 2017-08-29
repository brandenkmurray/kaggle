library(readr)
library(data.table)
library(caret)
setwd("/media/branden/SSHD1/kaggle/bnp")

train <- data.table(read.csv("./train.csv"))

# 5 fold
set.seed(2016)
cvFoldsList <- createFolds(train$target, k=5, list=TRUE, returnTrain=FALSE)
# TrainList for use with Caret
set.seed(2016)
cvFoldsTrainList <- createFolds(train$target, k=5, list=TRUE, returnTrain=TRUE)
set.seed(2016)
cvFolds <- data.frame(foldIndex=createFolds(train$target, k=5, list=FALSE))

save(cvFoldsList, file="./data_trans/cvFoldsList.rda")
save(cvFoldsTrainList, file="./data_trans/cvFoldsTrainList.rda")
write_csv(cvFolds, "./data_trans/cvFolds.csv")

# 10 fold
set.seed(2016)
cvFoldsList10 <- createFolds(train$target, k=10, list=TRUE, returnTrain=FALSE)
# TrainList for use with Caret
set.seed(2016)
cvFoldsTrainList10 <- createFolds(train$target, k=10, list=TRUE, returnTrain=TRUE)
set.seed(2016)
cvFolds10 <- data.frame(foldIndex=createFolds(train$target, k=10, list=FALSE))

save(cvFoldsList10, file="./data_trans/cvFoldsList10.rda")
save(cvFoldsTrainList10, file="./data_trans/cvFoldsTrainList10.rda")
write_csv(cvFolds10, "./data_trans/cvFolds10.csv")

# layer 2 folds
set.seed(1234)
cvFoldsList_lay2 <- createFolds(train$target, k=6, list=TRUE, returnTrain=FALSE)
set.seed(1234)
cvFoldsTrainList_lay2 <- createFolds(train$target, k=6, list=TRUE, returnTrain=TRUE)
set.seed(1234)
cvFolds_lay2 <- data.frame(foldIndex=createFolds(train$target, k=6, list=FALSE))

save(cvFoldsList_lay2, file="./data_trans/cvFoldsList_lay2.rda")
save(cvFoldsTrainList_lay2, file="./data_trans/cvFoldsTrainList_lay2.rda")
write_csv(cvFolds_lay2, "./data_trans/cvFolds_lay2.csv")
