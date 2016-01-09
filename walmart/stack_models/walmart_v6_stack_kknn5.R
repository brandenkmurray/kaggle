library(readr)
library(data.table)
library(caret)
library(reshape2)
library(dplyr)
library(kknn)
setwd("/home/branden/Documents/kaggle/walmart")

ts1Trans <- data.table(read_csv("./data_trans/ts1Trans3_prop_simil.csv", col_types=paste(replicate(6839, "n"), collapse = "")))

# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
load("./data_trans/cvFoldsList.rda")

# Only do KNN with summary variables and Departments
kknnPre <- preProcess(ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), pcaComp=200)
t1kknn <- predict(kknnPre, ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE])
# s1knn <- data.frame(matrix(rep(0, ncol(t1knn)*nrow(s1Trans)), ncol=ncol(t1knn), nrow=nrow(s1Trans)))
# colnames(s1knn) <- names(t1knn)
# s1knn[,colnames(s1knn) %in% colnames(t1knn)] <- s1Trans[,colnames(s1Trans) %in% colnames(s1knn),with=FALSE]
s1kknn <- predict(kknnPre, ts1Trans[filter==2,4:ncol(ts1Trans), with=FALSE])

t1kknn$class <- ts1Trans[filter==0, "class",with=FALSE]
t1kknn$class <- as.factor(t1kknn$class)


kknn5_stack_preds <- matrix(0, nrow=nrow(t1kknn), ncol=38)
logLossTable <- data.frame(fold=seq(1:length(cvFoldsList)), LogLoss=rep(0, length(cvFoldsList)))
for (i in 1:length(cvFoldsList)){
  kknn5 <- kknn(as.factor(class) ~ .,
                train=t1kknn[-cvFoldsList[[i]],], 
                test=t1kknn[cvFoldsList[[i]],], 
                k=300, 
                distance=1,
                kernel="triweight")
  kknn5_stack_preds[-cvFoldsList[[i]],] <- kknn5$prob
  logLossTable[i,2] <- LogLoss(model.matrix(~class-1, t1kknn[-cvFoldsList[[i]],"class", with=FALSE]), kknn5$prob)
}

cvPreds <- kknn5_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("kknn5", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"VisitNumber",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_kknn5.csv", row.names=FALSE)

# LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[5]],"class", with=FALSE]), kknn5_stack_preds$prob)
# LogLoss(model.matrix(~class-1, t1kknn[,"class", with=FALSE]), kknn5_stack_preds)


kknn5full <- kknn(as.factor(class) ~ .,
              train=t1kknn, 
              test=s1kknn, 
              k=300, 
              distance=1,
              kernel="triweight")
save(kknn5full, file="./stack_models/kknn5full.rda")

testPreds <- kknn5full$prob
colnames(testPreds) <- cnames
testPreds <- cbind(ts1Trans[filter==2,"VisitNumber",with=FALSE], testPreds)
write.csv(testPreds, "./stack_models/testPreds_kknn5full.csv", row.names=FALSE)
