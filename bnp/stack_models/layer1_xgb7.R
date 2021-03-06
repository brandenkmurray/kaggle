library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
# setwd("/media/branden/SSHD1/kaggle/bnp")
setwd("~/bnp")
ts1Trans <- data.table(read.csv("./data_trans/ts1Trans_v2.csv"))
# xgbImpVars <- data.table(read_csv("./stack_models/xgb7Imp.csv"))
load("./data_trans/cvFoldsList.rda")
varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter"), with=FALSE]))

# set.seed(201601)
# tsne_feats <- Rtsne(data.matrix(ts1Trans[,varnames,with=FALSE]), dims=2, initial_dims = 200, perplexity=80, theta=0.1, check_duplicates=TRUE, max_iter=500, verbose=TRUE)
# tsne_Y <- as.data.frame(tsne_feats$Y)
# colnames(tsne_Y) <- c("tsne_1", "tsne_2")
# write.csv(tsne_Y, "./stack_models/tsne_xgb4.csv", row.names=FALSE)
# tsne_Y$target <- as.factor(make.names(ts1Trans$target))
# (gg <- ggplot(tsne_Y[ts1Trans$filter==0,], aes(x=tsne_1, y=tsne_2, colour=target)) + geom_point(size=1))
# 
# ts1Trans <- cbind(ts1Trans, tsne_Y[,1:2])

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))

grid <- expand.grid(eta=c(0.003,0.007,0.01,0.013,0.017,0.02,0.025,0.03,0.04,0.05), md=seq(1,17,1), mcw=c(rep(1,50),seq(2,100,2)), ss=seq(.25,1,.05), csbt=seq(.25,1,.05))
set.seed(44)
models <- 100
samp <- sort(sample(nrow(grid), models, replace = FALSE))
(grid_samp <- grid[samp,])

bagPreds <- matrix(0,nrow=nrow(ts1Trans[ts1Trans$filter==0,]), ncol=nrow(grid_samp))
stopRounds <- data.frame(iter=rownames(grid_samp), round=0)
#Load submission file for IDs and column labels
sub <- read.csv("./sample_submission.csv")
# Model using the entire training set
bagPreds_test <- matrix(0, nrow=nrow(ts1Trans[ts1Trans$filter==2,]), ncol=nrow(grid_samp))
for (i in 1:nrow(grid_samp)) {
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
  xgb7cv <- xgb.cv(data=dtrain,
                    params=param,
                    nrounds=50000,
                    print.every.n=200,
                    folds=cvFoldsList,
                    prediction=TRUE,
                    early.stop.round=200,
                    maximize=FALSE)
  Sys.time() - tme
  bagPreds[,i] <- xgb7cv$pred
  min(xgb7cv$dt$test.logloss.mean)
  stopRounds[i,"round"] <- which.min(xgb7cv$dt$test.logloss.mean)
  write_csv(stopRounds, "./stack_models/stopRounds_xgb7cv.csv")
  
  #Move the CSV to inside the loop in case there's a crash midway through - Can continue loop from where it left off
  cvPreds <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], xgb7cv_pred=bagPreds)
  colnames(cvPreds) <- c("ID", paste0("cvPreds_xgb7_mod",1:models))
  write.csv(cvPreds, "./stack_models/cvPreds/cvPreds_xgb7.csv", row.names=FALSE) 
 
  set.seed(2015+i)
  xgb7 <- xgb.train(data=dtrain,
                    params=param,
                    nrounds=round(stopRounds[i,2]*1.125))
  bagPreds_test[,i] <- predict(xgb7, newdata=data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
  
  submission <- data.frame(ID=sub$ID, PredictedProb=bagPreds_test)
  colnames(submission) <- c("ID", paste0("testPreds_xgb7_mod",1:models))
  write.csv(submission, "./stack_models/testPreds/testPreds_xgb7.csv", row.names=FALSE)
}


# #Load submission file for IDs and column labels
# sub <- read.csv("./sample_submission.csv")
# # Model using the entire training set
# bagPreds_test <- matrix(0, nrow=nrow(ts1Trans[ts1Trans$filter==2,]), ncol=nrow(grid_samp))
# for (i in 1:nrow(grid_samp)) {
#   print(data.frame(i, grid_samp[i,]))
#   param <- list(booster="gbtree",
#                 eta=grid_samp[i,"eta"],
#                 max_depth=grid_samp[i,"md"],
#                 min_child_weight=grid_samp[i,"mcw"],
#                 subsample=grid_samp[i,"ss"],
#                 colsample_bytree=grid_samp[i,"csbt"],
#                 objective="binary:logistic",
#                 eval_metric="logloss")
#   
#   set.seed(2015+i)
#   xgb7 <- xgb.train(data=dtrain,
#                      params=param,
#                      nrounds=round(stopRounds[i,2]*1.125))
#   bagPreds_test[,i] <- predict(xgb7, newdata=data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
#   
#   submission <- data.frame(ID=sub$ID, PredictedProb=bagPreds_test)
#   colnames(submission) <- c("ID", paste0("testPreds_xgb7_mod",1:models))
#   write.csv(submission, "./stack_models/testPreds/testPreds_xgb7.csv", row.names=FALSE)
# }

submission2 <- data.frame(ID=sub$ID, PredictedProb=rowMeans(bagPreds_test))
write.csv(submission2, "./stack_models/testPreds/testPreds_means/testPreds_xgb7_means.csv", row.names=FALSE)

