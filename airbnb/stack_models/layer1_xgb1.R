library(xgboost)
library(data.table)
library(readr)

setwd("/home/branden/Documents/kaggle/airbnb")
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v1.csv"))

load("./data_trans/cvFoldsList.rda")

varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))
# dval <- xgb.DMatrix(data=data.matrix(train[cvFoldsList[[1]],varnames, with=FALSE]),label=data.matrix(train$class[cvFoldsList[[1]]]))
# dtrain <- xgb.DMatrix(data=data.matrix(train[-cvFoldsList[[1]],varnames, with=FALSE]),label=data.matrix(train$class[-cvFoldsList[[1]]]))
# watchlist <- list(train=dtrain, val=dval)

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0,varnames, with=FALSE]),label=data.matrix(ts1Trans$class[ts1Trans$filter==0]))

param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=12,
              eta = .05,
              max_depth=3,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              seed=201512
)


(tme <- Sys.time())
xgb1 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 5000,
               folds=cvFoldsList,
               maximize=FALSE,
               prediction=TRUE,
               print.every.n = 20,
               early.stop.round=50)
Sys.time() - tme
save(xgb1, file="./stack_models/xgb1.rda")

cvPreds <- xgb1$pred
classMap <- read_csv("./data_trans/classMap.csv")
cnames <- paste("xgb1", classMap$country_destination, sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1Trans[filter==0,"id",with=FALSE], cvPreds), "./stack_models/cvPreds_xgb1.csv", row.names=FALSE) 

rounds <- floor(which.min(xgb1$dt$test.mlogloss.mean) * 1.15)

(tme <- Sys.time())
xgb1full <- xgb.train(data = dtrain,
                       params = param,
                       nrounds = rounds,
                       maximize=FALSE,
                       print.every.n = 20)
Sys.time() - tme
save(xgb1full, file="./stack_models/xgb1full.rda")


testPreds <- predict(xgb1full, data.matrix(ts1Trans[filter==2,varnames, with=FALSE]))
testPreds <- as.data.frame(matrix(testPreds, nrow=12))
classMap <- read_csv("./data_trans/classMap.csv")
rownames(testPreds) <- classMap$country_destination
write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb1.csv", row.names=FALSE)
testPreds_top5 <- as.vector(apply(testPreds, 2, function(x) names(sort(x)[12:8])))


# create submission 
idx = ts1Trans$id[ts1Trans$filter==2]
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/testPreds_xgb1.csv", quote=FALSE, row.names = FALSE)
