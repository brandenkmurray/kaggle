library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(gtools)
setwd("/home/branden/Documents/kaggle/walmart")

LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

# Load CV predictions from models
load("./stack_models/xgbLay2_v14.rda")
load("./stack_models/xgbLay2_v15.rda")
nnLay2_v1 <- read_csv("./stack_models/cvPreds_lay2_nn1.csv")

nnLay2_v1$VisitNumber <- NULL

t1 <- data.table(read_csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class)]

set.seed(76)
cvFoldsList_lay3 <- createFolds(t1$class, k=6, list=TRUE, returnTrain=FALSE)

x <- seq(.6,.7,0.01)

blend_preds <- as.data.frame(matrix(0, nrow=nrow(train), ncol=38)) 

# for (p in x){
#   for (i in 1:length(cvFoldsList_lay3)){
#     tmp <- xgbLay2_v15$pred[cvFoldsList_lay3[[i]]]^p * nnLay2_v1[cvFoldsList_lay3[[i]]]^(1-p)
#   
#   }
# }
act <- as.data.frame(model.matrix(~as.factor(class)-1, t1[,"class", with=FALSE]))
act <- act[,mixedorder(colnames(act))]
logLossTable[i,2] <- LogLoss(act, tmp)

logLossTable <- data.frame(x=x, LogLoss=rep(0, length(x)))

i=1
for (p in x){
  tmp <- xgbLay2_v15$pred*p + nnLay2_v1*(1-p)
  logLossTable[i,2]  <- LogLoss(act, tmp)
  i = i + 1
}



# Load Test Set predictions from models trained on the entire training set
xgbLay2_v14_fullpreds <- read_csv("./stack_models/xgbLay2_v14preds.csv")
xgbLay2_v15_fullpreds <- read_csv("./stack_models/xgbLay2_v15preds.csv")
nnLay2_v1_fullpreds <- read_csv("./stack_models/nnlay2_v1preds.csv")

# Edit and bind test set predictions
xgbLay2_v14_fullpreds$VisitNumber <- NULL
xgbLay2_v15_fullpreds$VisitNumber <- NULL
nnLay2_v1_fullpreds$VisitNumber <- NULL

# lay1fullpreds <- cbind(xgb1fullpreds,xgb7fullpreds, kknn1fullpreds, nn1fullpreds, nn2fullpreds)
# Predict the test set using the XGBOOST stacked model

blendedPreds <- xgbLay2_v15_fullpreds*.67 + nnLay2_v1_fullpreds*(1-.67)


samp <- read_csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(blendedPreds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, blendedPreds)
write.csv(submission, "./stack_models/blendedPreds_v2.csv", row.names=FALSE)
