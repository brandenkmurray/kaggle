library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v25.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList10.rda")


varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))
dtest <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
rm(ts1Trans); gc()
  
param <- list(objective="binary:logistic",
                eval_metric="logloss",
                eta = .01,
                max_depth=7,
                min_child_weight=1,
                subsample=.8,
                colsample_bytree=.4
                # nthread=13
  )
  
set.seed(201512)
(tme <- Sys.time())
xgb38cv <- xgb.cv(data = dtrain,
                   params = param,
                   nrounds = 40000,
                   folds=cvFoldsList10,
                   maximize=FALSE,
                   prediction=TRUE,
                   print.every.n = 50,
                   early.stop.round=200)
Sys.time() - tme
save(xgb38cv, file="./stack_models/xgb38cv.rda")

write.csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], PredictedProb=xgb38cv$pred), "./stack_models/cvPreds/cvPreds_xgb38.csv", row.names=FALSE)

minLossRound <- which.min(xgb38cv$dt$test.logloss.mean)
rounds <- floor(minLossRound * 1.08)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
watchlist <- list(train=dtrain)
set.seed(201513)
(tme <- Sys.time())
xgb38full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = 500,
                      maximize=FALSE,
                      watchlist=watchlist,
                      print.every.n = 50)
Sys.time() - tme
save(xgb38full, file="./stack_models/xgb38full.rda")

preds <- predict(xgb38full, dtest)
xgb35preds <- read.csv("./stack_models/testPreds/testPreds_xgb35.csv")
submission <- data.frame(ID=xgb35preds$ID, PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb38.csv", row.names=FALSE)


xgb38Imp <- xgb.importance(feature_names = varnames, model=xgb38full)
write.csv(xgb38Imp, "./stack_models/xgb38Imp.csv", row.names=FALSE)


ptrain <- predict(xgb38full, dtrain, outputmargin=TRUE)
ptest  <- predict(xgb38full, dtest, outputmargin=TRUE)
# set the base_margin property of dtrain and dtest
# base margin is the base prediction we will boost from
setinfo(dtrain, "base_margin", ptrain)
setinfo(dtest, "base_margin", ptest)

