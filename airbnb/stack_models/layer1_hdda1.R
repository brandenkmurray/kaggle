library(readr)
library(data.table)
library(caret)
library(reshape2)
library(dplyr)
library(kknn)
library(Matrix)
library(doParallel)
setwd("/home/branden/Documents/kaggle/airbnb")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v4.csv"))
xgbImpVars <- read_csv("./stack_models/xgb7Imp.csv")
load("./data_trans/cvFoldsList.rda")
#ndcg metric from air's script
ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = 12
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}



# Only do KNN with summary variables and Departments
kknnPre <- preProcess(ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE], method=c("zv","center","scale"))
t1kknn <- predict(kknnPre, ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE])
# s1knn <- data.frame(matrix(rep(0, ncol(t1knn)*nrow(s1Trans)), ncol=ncol(t1knn), nrow=nrow(s1Trans)))
# colnames(s1knn) <- names(t1knn)
# s1knn[,colnames(s1knn) %in% colnames(t1knn)] <- s1Trans[,colnames(s1Trans) %in% colnames(s1knn),with=FALSE]
s1kknn <- predict(kknnPre, ts1Trans[filter==2,4:ncol(ts1Trans), with=FALSE])

t1kknn$class <- ts1Trans[filter==0, "class",with=FALSE]
t1kknn$class <- as.factor(t1kknn$class)


kknn4_stack_preds <- matrix(0, nrow=nrow(t1kknn), ncol=12)
logLossTable <- data.frame(fold=seq(1:length(cvFolds)), LogLoss=rep(0, length(cvFolds)))
for (i in 1:length(cvFolds)){
  kknn4 <- kknn(as.factor(class) ~ .,
                train=t1kknn[cvFolds[[i]],], 
                test=t1kknn[-cvFolds[[i]],], 
                k=300, 
                distance=1,
                kernel="triweight")
  kknn4_stack_preds[-cvFolds[[i]],] <- kknn4$prob
  logLossTable[i,2] <- LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[i]],"class", with=FALSE]), kknn4$prob)
}

cvPreds <- kknn4_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("kknn4", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"VisitNumber",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_kknn4.csv", row.names=FALSE)

# LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[5]],"class", with=FALSE]), kknn4_stack_preds$prob)
# LogLoss(model.matrix(~class-1, t1kknn[,"class", with=FALSE]), kknn4_stack_preds)


kknn4full <- kknn(as.factor(class) ~ .,
              train=t1kknn, 
              test=s1kknn, 
              k=300, 
              distance=1,
              kernel="triweight")
save(kknn4full, file="./stack_models/kknn4full.rda")

testPreds <- kknn4full$prob
colnames(testPreds) <- cnames
testPreds <- cbind(ts1Trans[filter==2,"VisitNumber",with=FALSE], testPreds)
write.csv(testPreds, "./stack_models/testPreds_kknn4full.csv", row.names=FALSE)
