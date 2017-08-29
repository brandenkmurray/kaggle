library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v28.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList.rda")


varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))

  
param <- list(objective="binary:logistic",
                eval_metric="logloss",
                eta = .01,
                max_depth=7,
                min_child_weight=1,
                subsample=.8,
                colsample_bytree=.4,
                nthread=13
  )
  
set.seed(201512)
(tme <- Sys.time())
xgb41cv <- xgb.cv(data = dtrain,
                   params = param,
                   nrounds = 8000,
                   folds=cvFoldsList,
                   maximize=FALSE,
                   prediction=TRUE,
                   print.every.n = 50,
                   early.stop.round=200)
Sys.time() - tme
save(xgb41cv, file="./stack_models/xgb41cv.rda")

write.csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], PredictedProb=xgb41cv$pred), "./stack_models/cvPreds/cvPreds_xgb41.csv", row.names=FALSE)

minLossRound <- which.min(xgb41cv$dt$test.logloss.mean)
rounds <- floor(minLossRound * 1.0)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201512)
(tme <- Sys.time())
xgb41full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = rounds,
                      maximize=FALSE,
                      print.every.n = 20)
Sys.time() - tme
save(xgb41full, file="./stack_models/xgb41full.rda")

preds <- predict(xgb41full, data.matrix(ts1Trans[filter==2, c(varnames), with=FALSE]))
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb41.csv", row.names=FALSE)


xgb41Imp <- xgb.importance(feature_names = colnames(ts1Trans[filter==0, c(varnames), with=FALSE]), model=xgb41full)
write.csv(xgb41Imp, "./stack_models/xgb41Imp.csv", row.names=FALSE)
