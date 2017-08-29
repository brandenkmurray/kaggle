library(readr)
library(data.table)
# Controls the maximum RAM that can be used with extraTrees
options(java.parameters="-Xmx128g")
library(extraTrees)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(gtools)
library(doParallel)
setwd("/media/branden/SSHD1/kaggle/bnp")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)

#################
# Logloss function
#################
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}
#################

ts1Trans <- fread("./data_trans/ts2Trans_v18.csv")
load("./data_trans/cvFoldsList.rda")
# xgb6cv <- read_csv("./stack_models/cvPreds_xgb6.csv")
# Create a vector the variable names to be used
varnames <- names(ts1Trans[filter==0, 5:ncol(ts1Trans), with=FALSE])

# Was only necessary for easier filtering of the validation set
# train <- ts1Trans[filter==0, c("target",varnames), with=FALSE]
# train$target <- as.factor(make.names(train$target))
# test <- ts1Trans[filter==2, c("target",varnames), with=FALSE]


pp <- preProcess(ts1Trans[filter==0,varnames,with=FALSE], method=c("pca"), thresh=0.99)
train <- predict(pp, ts1Trans[filter==0,varnames,with=FALSE])
test <- predict(pp, ts1Trans[filter==2,varnames,with=FALSE])

# Set ET params
nt <- 200
m <- 103
nodes <- 1
cuts <- 100


et2_stack_preds <- data.frame(cvPred=matrix(0, nrow=nrow(train), ncol=1)) 
logLossTable <- data.frame(fold=seq(1:length(cvFoldsList)), LogLoss=rep(0, length(cvFoldsList)))
bags <- 1
(tme <- Sys.time())
for (j in 1:bags){
  for (i in 1:length(cvFoldsList)){
    set.seed(201512+j)
    # set.seed(201512)
    et2 <- extraTrees(x=data.matrix(train[-cvFoldsList[[i]],-c("target"), with=FALSE]),
                    y=train$target[-cvFoldsList[[i]]],
                    ntree=nt,
                    mtry=m,
                    nodesize=nodes,
                    numRandomCuts=cuts,
                    numThreads=threads
    )
       et2_pred <- predict(et2, newdata=data.matrix(train[cvFoldsList[[i]],-c("target"), with=FALSE]), probability=TRUE)
       LogLoss(model.matrix(~target-1, train[cvFoldsList[[i]],"target", with=FALSE])[,2], et2_pred[,2])
  
#       et2_stack_preds[cvFoldsList[[i]],] <- predict(et2, newdata=data.matrix(train[cvFoldsList[[i]],varnames, with=FALSE]), probability=TRUE)
    tmp <- as.data.frame(predict(et2, newdata=data.matrix(train[cvFoldsList[[i]],-c("target"), with=FALSE]), probability=TRUE))
    # doing "+ tmp" is for bagging purposes
    et2_stack_preds[cvFoldsList[[i]],] <- et2_stack_preds[cvFoldsList[[i]],] + tmp[,2]
    actTmp <- model.matrix(~target-1, train[cvFoldsList[[i]],"target", with=FALSE])[,2]
    logLossTable[i,2] <- logLossTable[i,2] + LogLoss(actTmp, et2_stack_preds[cvFoldsList[[i]],])
    rm(et2, tmp)
    gc()
  }
}
Sys.time() - tme
et2_stack_preds_adj <- et2_stack_preds/bags
logLossTable[,2] <- logLossTable[,2]/bags

cvPreds <- et2_stack_preds
samp <- read.csv('sample_submission.csv')
colnames(cvPreds) <- "PredictedProb_et2"
cvPreds <- cbind(ts1Trans[filter==0,"ID",with=FALSE], cvPreds)
write_csv(cvPreds, "./stack_models/cvPreds/cvPreds_et2.csv")

## Create a model using the full dataset -- make predictions on test set for use in future stacking
preds <- data.frame(testPred=matrix(0, nrow=nrow(test), ncol=1))  
for (j in 1:bags){
  set.seed(201512+j)
  (tme <- Sys.time())
  et2full <- extraTrees(x=data.matrix(train[,varnames, with=FALSE]),
                        y=train$target,
                        ntree=nt,
                        mtry=m,
                        nodesize=nodes,
                        numRandomCuts=cuts,
                        numThreads=threads
  )
  tmp <- predict(et2full, data.matrix(test[,varnames, with=FALSE]), probability=TRUE)
  preds <- preds + tmp[,2]
}
Sys.time() - tme
preds_adj <- preds/bags



samp <- read.csv('sample_submission.csv')
names(preds_adj) <- "PredictedProb"
submission <- data.frame(ID=ts1Trans[filter==2,"ID",with=FALSE], PredictedProb=preds_adj)
write.csv(submission, "./stack_models/testPreds/testPreds_et2.csv", row.names=FALSE)

