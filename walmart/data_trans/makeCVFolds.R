library(readr)
library(data.table)
library(caret)
setwd("/home/branden/Documents/kaggle/walmart")


ts1Trans <- data.table(read_csv("./data_trans/ts1Trans3_prop.csv", col_types=paste(replicate(6839, "n"), collapse = "")))
set.seed(2016)
cvFoldsList <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=5, list=TRUE, returnTrain=FALSE)
set.seed(2016)
cvFolds <- data.frame(foldIndex=createFolds(ts1Trans$class[ts1Trans$filter==0], k=5, list=FALSE))

save(cvFoldsList, file="./data_trans/cvFoldsList.rda")
write_csv(cvFolds, "./data_trans/cvFolds.csv")

set.seed(1234)
cvFoldsList_lay2 <- createFolds(ts1Trans$class[ts1Trans$filter==0], k=4, list=TRUE, returnTrain=FALSE)
set.seed(1234)
cvFolds_lay2 <- data.frame(foldIndex=createFolds(ts1Trans$class[ts1Trans$filter==0], k=4, list=FALSE))

save(cvFoldsList_lay2, file="./data_trans/cvFoldsList_lay2.rda")
write_csv(cvFolds_lay2, "./data_trans/cvFolds_lay2.csv")
