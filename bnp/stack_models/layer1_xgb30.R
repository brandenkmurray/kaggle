library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
# setwd("/media/branden/SSHD1/kaggle/bnp")
setwd("~/ebs/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v18.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList.rda")

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))

grid <- expand.grid(eta=c(0.003,0.01,0.017,0.025), md=seq(3,13,1), mcw=c(rep(1,20),seq(2,20,2),30,40,50), ss=seq(0.6,1,0.05), csbt=seq(0.25,1,0.05))
set.seed(244)
models <- 10
samp <- sort(sample(nrow(grid), models, replace = FALSE))
(grid_samp <- grid[samp,])

bagPreds <- matrix(0,nrow=nrow(ts1Trans[ts1Trans$filter==0,]), ncol=nrow(grid_samp))
stopRounds <- data.frame(iter=rownames(grid_samp), round=0, minLoss=0)
#Load submission file for IDs and column labels
sub <- read.csv("./sample_submission.csv")
# Model using the entire training set
bagPreds_test <- matrix(0, nrow=nrow(ts1Trans[ts1Trans$filter==2,]), ncol=nrow(grid_samp))
for (i in 1:models) {
  print(data.frame(i, grid_samp[i,]))
  param <- list(booster="gbtree",
                eta=grid_samp[i,"eta"],
                max_depth=grid_samp[i,"md"],
                min_child_weight=grid_samp[i,"mcw"],
                subsample=grid_samp[i,"ss"],
                colsample_bytree=grid_samp[i,"csbt"],
                objective="binary:logistic",
                eval_metric="logloss")
  
  (tme <- Sys.time())
  set.seed(201510)
  xgb30cv <- xgb.cv(data=dtrain,
                    params=param,
                    nrounds=20000,
                    print.every.n=200,
                    folds=cvFoldsList,
                    prediction=TRUE,
                    early.stop.round=200,
                    maximize=FALSE)
  Sys.time() - tme
  bagPreds[,i] <- xgb30cv$pred
  min(xgb30cv$dt$test.logloss.mean)
  stopRounds[i,"round"] <- which.min(xgb30cv$dt$test.logloss.mean)
  stopRounds[i,"minLoss"] <- min(xgb30cv$dt$test.logloss.mean)
  write.csv(stopRounds, "./stack_models/stopRounds_xgb30cv.csv", row.names=FALSE)
  
  #Move the CSV to inside the loop in case there's a crash midway through - Can continue loop from where it left off
  cvPreds <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], xgb30cv_pred=bagPreds)
  colnames(cvPreds) <- c("ID", paste0("cvPreds_xgb30_mod",1:models))
  write.csv(cvPreds, "./stack_models/cvPreds/cvPreds_xgb30.csv", row.names=FALSE) 
  
  set.seed(2015+i)
  xgb30 <- xgb.train(data=dtrain,
                     params=param,
                     nrounds=round(stopRounds[i,2]*1.125))
  bagPreds_test[,i] <- predict(xgb30, newdata=data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
  
  submission <- data.frame(ID=sub$ID, PredictedProb=bagPreds_test)
  colnames(submission) <- c("ID", paste0("testPreds_xgb30_mod",1:models))
  write.csv(submission, "./stack_models/testPreds/testPreds_xgb30.csv", row.names=FALSE)
  
  rm(xgb30cv, xgb30)
  gc()
}





