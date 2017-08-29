library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(caret)
library(ggplot2)
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v29_holdout.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
# load("./data_trans/cvFoldsList.rda")
# Create Folds
set.seed(2015)
cvFoldsList <- createFolds(ts1Trans$target[ts1Trans$filter==0],k=5, list=TRUE)

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))
dval <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==1, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==1]))
watch <- list(val=dval, train=dtrain)
  
param <- list(objective="binary:logistic",
                eval_metric="logloss",
                eta = .01,
                max_depth=7,
                min_child_weight=1,
                subsample=.8,
                colsample_bytree=.4,
                nthread=13
  )
  
## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201512)
(tme <- Sys.time())
xgb42full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = 8000,
                      maximize=FALSE,
                      watchlist=watch,
                      print.every.n = 50,
                      early.stop.round=200)
Sys.time() - tme
save(xgb42full, file="./stack_models/xgb42full_holdout.rda")

preds <- predict(xgb42full, data.matrix(ts1Trans[filter==1, c(varnames), with=FALSE]))
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==1], PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb42_holdout.csv", row.names=FALSE)

preds <- predict(xgb42full, data.matrix(ts1Trans[filter==2, c(varnames), with=FALSE]))
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb42_holdout_test.csv", row.names=FALSE)
