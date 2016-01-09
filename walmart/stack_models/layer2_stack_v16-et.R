library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
options(java.parameters="-Xmx125g")
library(extraTrees)
library(gtools)
setwd("/home/branden/Documents/kaggle/walmart")

LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

# Load CV predictions from models
xgb1preds <- read_csv("./stack_models/cvPreds_xgb1.csv")
xgb2preds <- read_csv("./stack_models/cvPreds_xgb2.csv")
xgb3preds <- read_csv("./stack_models/cvPreds_xgb3.csv")
xgb7preds <- read_csv("./stack_models/cvPreds_xgb7.csv")
xgb8preds <- read_csv("./stack_models/cvPreds_xgb8.csv")
xgb9preds <- read_csv("./stack_models/cvPreds_xgb9.csv")
xgb10preds <- read_csv("./stack_models/cvPreds_xgb10.csv")
xgb11preds <- read_csv("./stack_models/cvPreds_xgb11.csv")
kknn1preds <- read_csv("./stack_models/cvPreds_kknn1.csv")
kknn2preds <- read_csv("./stack_models/cvPreds_kknn2.csv")
kknn3preds <- read_csv("./stack_models/cvPreds_kknn3.csv")
kknn4preds <- read_csv("./stack_models/cvPreds_kknn4.csv")
nn1preds <- read_csv("./stack_models/cvPreds_nn1.csv")
nn2preds <- read_csv("./stack_models/cvPreds_nn2.csv")
et1preds <- read_csv("./stack_models/cvPreds_et1.csv")
glmnet1preds <- read_csv("./stack_models/cvPreds_glmnet1.csv")
glmnet2preds <- read_csv("./stack_models/cvPreds_glmnet2.csv")
# Edit and bind predictions
xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
xgb3preds$VisitNumber <- NULL
xgb7preds$VisitNumber <- NULL
xgb8preds$VisitNumber <- NULL
xgb9preds$VisitNumber <- NULL
xgb10preds$VisitNumber <- NULL
xgb11preds$VisitNumber <- NULL
kknn1preds$VisitNumber <- NULL
kknn2preds$VisitNumber <- NULL
kknn3preds$VisitNumber <- NULL
kknn4preds$VisitNumber <- NULL
nn1preds$VisitNumber <- NULL
nn2preds$VisitNumber <- NULL
et1preds$VisitNumber <- NULL
glmnet1preds$VisitNumber <- NULL
glmnet2preds$VisitNumber <- NULL
glmnet1preds[is.na(glmnet1preds)] <- 0
lay1preds <- cbind(xgb1preds, xgb2preds, xgb3preds, xgb7preds, xgb8preds, xgb9preds, xgb10preds, xgb11preds, kknn1preds, kknn2preds, kknn3preds,kknn4preds, nn1preds, nn2preds, et1preds, glmnet1preds)
# lay1preds <- cbind(xgb1preds,xgb7preds, kknn1preds, nn1preds, nn2preds)
# Add the class column to the dataset
t1 <- data.table(read_csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class)]

lay1preds <- data.table(cbind(class=as.factor(make.names(t1$class)), lay1preds))

# Create a validation set
# set.seed(1234)
# h <- sample(nrow(lay1preds), 2000)
# # Create DMatrices
# dval <- xgb.DMatrix(data=data.matrix(lay1preds[h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[h,"class", with=FALSE]))
# dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[-h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[-h,"class", with=FALSE]))
# watchlist <- list(val=dval,train=dtrain)

load("./data_trans/cvFoldsList_lay2.rda")

# lay1preds Model
# xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.564680+0.003101
# xgb1, xgb2, xgb3, kknn1, nn1 = 0.567281+0.002696
# xgb1, xgb7, kknn1, nn1, nn2 = 0.562542+0.003631
# subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.559679+0.002955
# eta=0.02, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2, et1 = 0.556184+0.002734
# eta=0.02,maxdep=3, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, xgb8,xgb10,kknn1, nn1, nn2, et1,glmnet1 = 0.54421+0.003521
# eta=0.02,maxdep=3, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, xgb8,xgb10,xgb11,kknn1, kknn2,kknn3,kknn4,nn1, nn2, et1,glmnet1,glmnet2 = 
et1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(lay1preds), ncol=38)) 
colnames(et1_stack_preds) <- unique(lay1preds$class)
et1_stack_preds <- et1_stack_preds[,mixedorder(colnames(et1_stack_preds))]
logLossTable <- data.frame(fold=seq(1:length(cvFoldsList_lay2)), LogLoss=rep(0, length(cvFoldsList_lay2)))
(tme <- Sys.time())
for (i in 1:length(cvFoldsList_lay2)){
  set.seed(201510)
  et1 <- extraTrees(x=data.matrix(lay1preds[-cvFoldsList_lay2[[i]],-c("class"),with=FALSE]),
                    y=lay1preds$class[-cvFoldsList_lay2[[i]]],
                    ntree=3000,
                    mtry=10,
                    nodesize=20,
                    numRandomCuts=10,
                    numThreads=16
  )
  # et1_pred <- predict(et1, newdata=data.matrix(lay1preds[cvFoldsList_lay2[[1]],-c("class"), with=FALSE]), probability=TRUE)
  # LogLoss(model.matrix(~as.factor(class)-1, lay1preds[cvFoldsList_lay2[[1]],"class", with=FALSE]), et1_pred)
  #   
  # et1_stack_preds[-cvFoldsList_lay2[[i]],] <- predict(et1, newdata=data.matrix(lay1preds[-cvFoldsList_lay2[[i]],varnames, with=FALSE]), probability=TRUE)
  tmp <- as.data.frame(predict(et1, newdata=data.matrix(lay1preds[cvFoldsList_lay2[[i]],-c("class"), with=FALSE]), probability=TRUE))
  tmp <- tmp[,mixedorder(colnames(tmp))]
  et1_stack_preds[cvFoldsList_lay2[[i]],colnames(tmp)] <- tmp
  actTmp <- as.data.frame(model.matrix(~class-1, lay1preds[cvFoldsList_lay2[[i]],"class", with=FALSE]))
  actTmp <- actTmp[,mixedorder(colnames(actTmp))]
  logLossTable[i,2] <- LogLoss(actTmp, et1_stack_preds[cvFoldsList_lay2[[i]],])
}
Sys.time() - tme

cvPreds <- et1_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("lay1_et", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(VisitNumber=t1$VisitNumber, cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_lay2_et1.csv", row.names=FALSE)


# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read_csv("./stack_models/testPreds_xgb1full.csv")
xgb2fullpreds <- read_csv("./stack_models/testPreds_xgb2full.csv")
xgb3fullpreds <- read_csv("./stack_models/testPreds_xgb3full.csv")
xgb7fullpreds <- read_csv("./stack_models/testPreds_xgb7full.csv")
xgb8fullpreds <- read_csv("./stack_models/testPreds_xgb8full.csv")
xgb9fullpreds <- read_csv("./stack_models/testPreds_xgb9full.csv")
xgb10fullpreds <- read_csv("./stack_models/testPreds_xgb10full.csv")
xgb11fullpreds <- read_csv("./stack_models/testPreds_xgb11full.csv")
kknn1fullpreds <- read_csv("./stack_models/testPreds_kknn1full.csv")
kknn2fullpreds <- read_csv("./stack_models/testPreds_kknn2full.csv")
kknn3fullpreds <- read_csv("./stack_models/testPreds_kknn3full.csv")
kknn4fullpreds <- read_csv("./stack_models/testPreds_kknn4full.csv")
nn1fullpreds <- read_csv("./stack_models/testPreds_nn1full.csv")
nn2fullpreds <- read_csv("./stack_models/testPreds_nn2full.csv")
et1fullpreds <- read_csv("./stack_models/testPreds_et1full.csv")
glmnet1fullpreds <- read_csv("./stack_models/testPreds_glmnet1full.csv")
glmnet1fullpreds[is.na(glmnet1fullpreds)] <- 0
# Edit and bind test set predictions
xgb1fullpreds$VisitNumber <- NULL
xgb2fullpreds$VisitNumber <- NULL
xgb3fullpreds$VisitNumber <- NULL
xgb7fullpreds$VisitNumber <- NULL
xgb8fullpreds$VisitNumber <- NULL
xgb9fullpreds$VisitNumber <- NULL
xgb10fullpreds$VisitNumber <- NULL
xgb11fullpreds$VisitNumber <- NULL
kknn1fullpreds$VisitNumber <- NULL
kknn2fullpreds$VisitNumber <- NULL
kknn3fullpreds$VisitNumber <- NULL
kknn4fullpreds$VisitNumber <- NULL
nn1fullpreds$VisitNumber <- NULL
nn2fullpreds$VisitNumber <- NULL
et1fullpreds$VisitNumber <- NULL
glmnet1fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds, xgb3fullpreds, xgb7fullpreds, xgb8fullpreds, xgb9fullpreds,xgb10fullpreds, xgb11fullpreds, kknn1fullpreds,kknn2fullpreds,kknn3fullpreds,kknn4fullpreds, nn1fullpreds, nn2fullpreds, et1fullpreds, glmnet1fullpreds)
# lay1fullpreds <- cbind(xgb1fullpreds,xgb7fullpreds, kknn1fullpreds, nn1fullpreds, nn2fullpreds)
# Predict the test set using the XGBOOST stacked model

set.seed(201510)
et1 <- extraTrees(x=data.matrix(lay1preds[,-c("class"),with=FALSE]),
                  y=lay1preds$class,
                  ntree=4000,
                  mtry=10,
                  nodesize=20,
                  numRandomCuts=10,
                  numThreads=16
)
save(et1, file="./stack_models/xgbLay2_v16_et.rda")

lay2preds <- predict(et1, newdata=data.matrix(lay1fullpreds), probability=TRUE)
preds <- lay2preds[,mixedorder(colnames(lay2preds))]
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_v16preds-et.csv", row.names=FALSE)
