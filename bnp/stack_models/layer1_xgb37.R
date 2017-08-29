library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
# setwd("/media/branden/SSHD1/kaggle/bnp")
setwd("~/ebs/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v20.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList.rda")

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))

grid <- expand.grid(eta=c(0.003,0.01,0.017,0.025), md=seq(3,13,1), mcw=c(rep(1,20),seq(2,20,2),30,40,50), ss=seq(0.6,1,0.05), csbt=seq(0.25,1,0.05))
set.seed(245)
models <- 10
samp <- sort(sample(nrow(grid), models, replace = FALSE))
(grid_samp <- grid[samp,])


foldPreds <- matrix(0,nrow=nrow(ts1Trans[ts1Trans$filter==0,]), ncol=nrow(grid_samp))
testPreds <- matrix(0, nrow=nrow(ts1Trans[ts1Trans$filter==2,]), ncol=nrow(grid_samp))
#Load submission file for IDs and column labels
sub <- read.csv("./sample_submission.csv")
for (j in 1:models) {
  print(data.frame(j, grid_samp[j,]))
  param <- list(booster="gbtree",
                eta=grid_samp[j,"eta"],
                max_depth=grid_samp[j,"md"],
                min_child_weight=grid_samp[j,"mcw"],
                subsample=grid_samp[j,"ss"],
                colsample_bytree=grid_samp[j,"csbt"],
                objective="binary:logistic",
                eval_metric="logloss")
  
  for (i in 1:length(cvFoldsList)) {
    print(paste0("Fold ",i))
    dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][-cvFoldsList[[i]], c(varnames),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][-cvFoldsList[[i]]]))
    dval <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(varnames),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][cvFoldsList[[i]]]))
    watch <- list(val=dval, train=dtrain)
    
    set.seed(201603)
    (tme <- Sys.time())
    xgb37 <- xgb.train(data = dtrain,
                       params = param,
                       nrounds = 40000,
                       maximize=FALSE,
                       watchlist=watch,
                       print.every.n = 50,
                       early.stop.round=200)
    Sys.time() - tme
    
    foldPreds[cvFoldsList[[i]],j] <- predict(xgb37, data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(varnames),with=FALSE]))
    testPreds[,j] <- testPreds[,j] + predict(xgb37, data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
    
    rm(xgb37); gc()
    
  }
  
  testPreds[,j] <- testPreds[,j]/length(cvFoldsList)
  
  cvPreds <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], foldPreds)
  colnames(cvPreds) <- c("ID",paste0("cvPreds_xgb37_mod",1:models))
  write_csv(cvPreds, "./stack_models/cvPreds/cvPreds_xgb37.csv")
  
  submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=testPreds)
  colnames(submission) <- c("ID", paste0("testPreds_xgb37_mod",1:models))
  write.csv(submission, "./stack_models/testPreds/testPreds_xgb37.csv", row.names=FALSE)
  
}


