library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v18.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList.rda")


varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))

  
param <- list(objective="binary:logistic",
                eval_metric="logloss",
                eta = .01,
                max_depth=7,
                min_child_weight=1,
                subsample=.8,
                colsample_bytree=.4,
                nthread=13
  )

models=5
foldPreds <- matrix(0, nrow = nrow(ts1Trans[ts1Trans$filter==0,]), ncol=models)
testPreds <- matrix(0, nrow = nrow(ts1Trans[ts1Trans$filter==2,]), ncol=models)
for (j in 1:models){
  set.seed(2016+j)
  sampVars <- sample(varnames, floor(length(varnames)*0.8))
  for (i in 1:length(cvFoldsList)) {
    print(paste0("Fold ",i))
    dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][-cvFoldsList[[i]], c(sampVars),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][-cvFoldsList[[i]]]))
    dval <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(sampVars),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][cvFoldsList[[i]]]))
    watch <- list(val=dval, train=dtrain)
      
    set.seed(201603)
    (tme <- Sys.time())
    xgb29 <- xgb.train(data = dtrain,
                       params = param,
                       nrounds = 50000,
                       maximize=FALSE,
                       watchlist=watch,
                       print.every.n = 50,
                       early.stop.round=200)
    Sys.time() - tme
    
    foldPreds[cvFoldsList[[i]],j] <- predict(xgb29, data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(sampVars),with=FALSE]))
    testPreds[,j] <- testPreds[,j] + predict(xgb29, data.matrix(ts1Trans[filter==2, c(sampVars),with=FALSE]))
    
    rm(xgb29); gc()
    
  }
}

testPreds_adj <- testPreds/length(cvFoldsList)

# Individual model CV predictions
colnames(foldPreds) <- paste0("PredictedProb_xgb29_mod",1:models)
write_csv(data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], foldPreds), "./stack_models/cvPreds/cvPreds_xgb29_mods.csv")
# Average of CV predictions
foldPreds_avg <- apply(foldPreds, 1, mean)
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], PredictedProb=foldPreds_avg)
write_csv(submission, "./stack_models/cvPreds/cvPreds_xgb29.csv")


# Individual model predictions
colnames(testPreds_adj) <- paste0("PredictedProb_xgb29_mod",1:models)
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], testPreds_adj)
write_csv(submission, "./stack_models/testPreds/testPreds_xgb29_mods.csv")

# Average of predictions
testPreds_adj_avg <- apply(testPreds_adj, 1, mean)
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=testPreds_adj_avg)
write_csv(submission, "./stack_models/testPreds/testPreds_xgb29.csv")


