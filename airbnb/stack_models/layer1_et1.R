library(readr)
library(data.table)
options(java.parameters="-Xmx128g")
library(extraTrees)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(gtools)
library(doParallel)
setwd("/home/branden/Documents/kaggle/airbnb")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)

#################
# Logloss function
#################
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}
#################

ts1Trans <- data.table(read.csv("./data_trans/ts2Trans_v11.csv"))
load("./data_trans/cvFoldsList.rda")
# xgb6cv <- read_csv("./stack_models/cvPreds_xgb6.csv")
# Create a vector the variable names to be used
varnames <- names(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE])

# Was only necessary for easier filtering of the validation set
train <- ts1Trans[filter==0, c("target",varnames), with=FALSE]
train$target <- as.factor(make.names(train$target))
test <- ts1Trans[filter==2, c("target",varnames), with=FALSE]



et1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(train), ncol=12)) 
colnames(et1_stack_preds) <- unique(train$target)
et1_stack_preds <- et1_stack_preds[,mixedorder(colnames(et1_stack_preds))]
logLossTable <- data.frame(fold=seq(1:length(cvFoldsList)), LogLoss=rep(0, length(cvFoldsList)))
bags <- 1
(tme <- Sys.time())
for (j in 1:bags){
  for (i in 1:length(cvFoldsList)){
    set.seed(201512+j)
    # set.seed(201512)
    et1 <- extraTrees(x=data.matrix(train[-cvFoldsList[[i]],-c("target"), with=FALSE]),
                    y=train$target[-cvFoldsList[[i]]],
                    ntree=5000,
                    mtry=13,
                    nodesize=5,
                    numRandomCuts=50,
                    numThreads=threads
    )
#        et1_pred <- predict(et1, newdata=data.matrix(train[cvFoldsList[[1]],-c("target"), with=FALSE]), probability=TRUE)
#        LogLoss(model.matrix(~target-1, train[cvFoldsList[[1]],"target", with=FALSE]), et1_pred)
#   
#       et1_stack_preds[cvFoldsList[[i]],] <- predict(et1, newdata=data.matrix(train[cvFoldsList[[i]],varnames, with=FALSE]), probability=TRUE)
    tmp <- as.data.frame(predict(et1, newdata=data.matrix(train[cvFoldsList[[i]],-c("target"), with=FALSE]), probability=TRUE))
    tmp <- tmp[,mixedorder(colnames(tmp))]
    et1_stack_preds[cvFoldsList[[i]],colnames(tmp)] <- et1_stack_preds[cvFoldsList[[i]],colnames(tmp)] + tmp
    actTmp <- as.data.frame(model.matrix(~target-1, train[cvFoldsList[[i]],"target", with=FALSE]))
    actTmp <- actTmp[,mixedorder(colnames(actTmp))]
    logLossTable[i,2] <- logLossTable[i,2] + LogLoss(actTmp, et1_stack_preds[cvFoldsList[[i]],])
    rm(et1, tmp)
    gc()
  }
}
Sys.time() - tme
et1_stack_preds_adj <- et1_stack_preds/bags
logLossTable[,2] <- logLossTable[,2]/bags

cvPreds <- et1_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("et1", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"id",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_et1.csv", row.names=FALSE)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
preds <- as.data.frame(matrix(0, nrow=nrow(test), ncol=3)) 
colnames(preds) <- unique(train$target)
preds <- preds[,mixedorder(colnames(preds))]
for (j in 1:bags){
  set.seed(201512+j)
  (tme <- Sys.time())
  et1full <- extraTrees(x=data.matrix(train[,varnames, with=FALSE]),
                        y=train$target,
                        ntree=5400,
                        mtry=13,
                        nodesize=5,
                        numRandomCuts=50,
                        numThreads=threads
  )
  tmp <- predict(et1full, data.matrix(test[,varnames, with=FALSE]), probability=TRUE)
  tmp <- tmp[,mixedorder(colnames(tmp))]
  preds <- preds + tmp
}
Sys.time() - tme
preds_adj <- preds/bags



samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(id=ts1Trans[filter==2,"id",with=FALSE], preds_adj)
write.csv(submission, "./stack_models/testPreds_et1.csv", row.names=FALSE)

